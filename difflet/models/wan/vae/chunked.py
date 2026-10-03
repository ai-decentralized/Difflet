"""Fixed-size Wan VAE decode graphs with explicit causal convolution state.

The first latent produces one pixel frame; subsequent latents produce four
for the standard Wan VAE. The second latent has a distinct cache layout
(``Rep`` sentinels and one-frame histories), so it has its own trace.
"""

from __future__ import annotations

import torch

from .modeling_vae import WanVAEDecoderConfig, WanVAEDecoderModel


def cache_layouts(config: WanVAEDecoderConfig, latent_shape: tuple[int, ...]):
    """Infer the two incoming state layouts using shape-only meta execution.

Unused cache slots and the temporal-upsample ``Rep`` markers stay Python
constants. Only live tensors cross the compiled graph boundary.
"""
    layouts = []
    with torch.device("meta"), torch.no_grad():
        model = WanVAEDecoderModel(config).eval()
        z = torch.zeros(latent_shape)
        state = model._clear_cache()
        for _ in range(2):
            model.decoder(model.post_quant_conv(z), feat_cache=state, feat_idx=[0])
            layouts.append(tuple(tuple(x.shape) if isinstance(x, torch.Tensor) else x for x in state))
    return tuple(layouts)


class WanVAEChunkModel(WanVAEDecoderModel):
    """Decode one latent frame and return pixels plus the updated tensor state.

All three traces keep the original checkpoint key names and share weights.
Phase selection depends only on the input signature, never on tensor values.
"""

    def __init__(self, config: WanVAEDecoderConfig):
        super().__init__(config)
        # Spatial dimensions do not affect which slots are tensors or markers.
        self._layouts = cache_layouts(config, (1, config.z_dim, 1, 2, 2))
        self.state_slots = tuple(i for i, slot in enumerate(self._layouts[1]) if isinstance(slot, tuple))
        if sum(isinstance(x, tuple) for x in self._layouts[0]) == len(self.state_slots):
            raise ValueError("Chunked Wan VAE requires temporal upsampling")

    def forward(self, z: torch.Tensor, *state: torch.Tensor):
        # A rank-one placeholder denotes an empty history. After the first
        # call conv_in has one historical frame, and thereafter it has two.
        # All compiled buckets have the SAME argument/output count: NxD's
        # executor and output packer are traced from the first example.
        if state and len(state) != len(self.state_slots):
            raise ValueError(f"Unexpected Wan VAE state tensor count: {len(state)}")
        if not state or state[0].ndim == 1:
            cache = self._clear_cache()
        else:
            layout = self._layouts[0 if state[0].shape[2] == 1 else 1]
            cache = list(layout)
            for index, tensor in zip(self.state_slots, state):
                if isinstance(layout[index], tuple):
                    cache[index] = tensor
        pixels = self.decoder(self.post_quant_conv(z), feat_cache=cache, feat_idx=[0])
        return (pixels.clamp(-1, 1), *(cache[i] if isinstance(cache[i], torch.Tensor) else z.new_zeros((1,)) for i in self.state_slots))


def decode_chunks(decoder, latents: torch.Tensor) -> torch.Tensor:
    """Run a complete request, resetting causal state at each request boundary."""
    if latents.ndim != 5 or latents.shape[2] < 1:
        raise ValueError("Wan VAE expects nonempty (B, C, T, H, W) latents")
    state = ()
    frames = []
    for index in range(latents.shape[2]):
        result = decoder(latents[:, :, index:index + 1].contiguous(), *state)
        frames.append(result[0])
        state = tuple(result[1:])
    return torch.cat(frames, dim=2)


class WanVAESplitChunkModel(WanVAEChunkModel):
    """Split after temporal upsampling; decode the spatial tail one frame at a time."""

    def __init__(self, config):
        super().__init__(config)
        temporal_blocks = [
            i for i, block in enumerate(self.decoder.up_blocks)
            if block.upsamplers is not None and block.upsamplers[0].mode == "upsample3d"
        ]
        self.split_block = temporal_blocks[-1] + 1
        if self.split_block == len(self.decoder.up_blocks):
            raise ValueError("Split Wan VAE needs a spatial tail after temporal upsampling")
        with torch.device("meta"), torch.no_grad():
            probe = WanVAEDecoderModel(config).eval()
            cache = probe._clear_cache()
            index = [0]
            probe.decoder(
                probe.post_quant_conv(torch.zeros(1, config.z_dim, 1, 2, 2)),
                feat_cache=cache, feat_idx=index, end_block=self.split_block,
            )
        self.tail_cache_start = index[0]
        self.head_slots = tuple(i for i in self.state_slots if i < self.tail_cache_start)
        self.tail_slots = tuple(i for i in self.state_slots if i >= self.tail_cache_start)
        self.state_count = max(len(self.head_slots), len(self.tail_slots))

    def forward(self, x, *state):
        slots = self.head_slots if x.ndim == 5 else self.tail_slots
        if state and len(state) != self.state_count:
            raise ValueError("Unexpected split Wan VAE state tensor count")
        if not state or state[0].ndim == 1:
            cache = self._clear_cache()
        else:
            layout = self._layouts[0 if state[0].shape[2] == 1 else 1]
            cache = list(layout)
            for index, tensor in zip(slots, state):
                if isinstance(layout[index], tuple):
                    cache[index] = tensor if tensor.ndim > 1 else None
        # The private tail input has an extra singleton axis, making routing
        # unambiguous even when latent and intermediate channel counts match.
        if x.ndim == 5:
            output = self.decoder(
                self.post_quant_conv(x), feat_cache=cache, feat_idx=[0],
                end_block=self.split_block,
            )
        else:
            output = self.decoder(
                x.squeeze(2), feat_cache=cache, feat_idx=[self.tail_cache_start],
                start_block=self.split_block,
            ).clamp(-1, 1)
        tensors = [cache[i] if isinstance(cache[i], torch.Tensor) else x.new_zeros((1,)) for i in slots]
        tensors.extend(x.new_zeros((1,)) for _ in range(self.state_count - len(slots)))
        return (output, *tensors)


def decode_split_chunks(decoder, latents):
    """Keep both partitions' causal state local to this request."""
    if latents.ndim != 5 or latents.shape[2] < 1:
        raise ValueError("Wan VAE expects nonempty (B, C, T, H, W) latents")
    head_state = tail_state = ()
    frames = []
    for index in range(latents.shape[2]):
        result = decoder(latents[:, :, index:index + 1].contiguous(), *head_state)
        features, head_state = result[0], tuple(result[1:])
        for frame in range(features.shape[2]):
            tail_input = features[:, :, frame:frame + 1].unsqueeze(2).contiguous()
            result = decoder(tail_input, *tail_state)
            frames.append(result[0])
            tail_state = tuple(result[1:])
    return torch.cat(frames, dim=2)


def split_input_shapes(config, latent_shape):
    """Discover all partition/phase signatures by a shape-only three-latent decode."""
    signatures = []
    with torch.device("meta"), torch.no_grad():
        model = WanVAESplitChunkModel(config).eval()

        def record(x, *state):
            state = state or tuple(x.new_zeros((1,)) for _ in range(model.state_count))
            signature = tuple(tuple(t.shape) for t in (x, *state))
            if signature not in signatures:
                signatures.append(signature)
            return model(x, *state)

        batch, channels, _frames, height, width = latent_shape
        decode_split_chunks(record, torch.zeros(batch, channels, 3, height, width))
    return signatures
