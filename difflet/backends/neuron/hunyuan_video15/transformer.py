"""Neuron application for HunyuanVideo 1.5's DiT: diffusers' ``HunyuanVideo15Transformer3DModel``.

The module is diffusers' own (no Difflet port exists; the Trainium path wraps the same class), built
on ``meta`` and loaded from diffusers' checkpoint key for key, fp32 -> the application dtype; in
compile mode its 54 identical ``transformer_blocks`` are compiled in place into one graph.

diffusers' forward runs as is: its text-token reorder (valid tokens of the image, ByT5 and
Qwen2.5-VL streams first, padding zeroed after) is data-dependent but fixed-length (729 + 256 + 1000
= 1,985 tokens), and it runs in the un-compiled top-level forward, so the compiled blocks see one
shape. (The Trainium path forks it because it compiles the whole model as one graph.)

Parallelism: TP1 (the 8.3B DiT is 15.5 GiB in bf16, one core), optionally with context
parallelism over the image tokens (Ulysses): every rank holds the whole DiT; the image stream is
split evenly across the CP group before the blocks and gathered after them, the text stream (729 +
256 + 1000 tokens) stays replicated, and the blocks' attention processor is replaced by one that
calls ``joint_ulysses_attention`` with the text's padding expressed as ``key_valid_len`` (diffusers
reorders valid text tokens first). That is exact for every valid query: diffusers' symmetric mask
and a key mask differ only on padded text queries, whose outputs are never used.
Numerics depend on two backend workarounds installed by ``prepare_runtime``: the bool bitwise guard
(the token refiner's mask) and fp32 LayerNorm for bf16 inputs.
"""

from __future__ import annotations

import json
import os

import torch

from difflet.backends.neuron.core.application_base import TorchNeuronApplicationBase

#: Spatial and temporal compression of HunyuanVideo 1.5's VAE (latent = (F-1)/4+1, H/16, W/16).
VAE_SPATIAL, VAE_TEMPORAL = 16, 4


class NeuronHunyuanVideo15TransformerApplication(TorchNeuronApplicationBase):
    """HunyuanVideo 1.5's DiT at one static shape (TP1)."""

    block_attrs = ("transformer_blocks",)

    def __init__(self, *, model_path, height: int = 480, width: int = 848, num_frames: int = 121,
                 text_seq_len: int = 1000, text_2_seq_len: int = 256, image_tokens: int = 729,
                 batch_size: int = 1, parallel=None, dtype=torch.bfloat16,
                 exec_mode: str | None = None, device="neuron", **kwargs) -> None:
        super().__init__(model_path=model_path, parallel=parallel, dtype=dtype,
                         exec_mode=exec_mode, device=device, **kwargs)
        if int(self.parallel.tp_degree) != 1:
            raise NotImplementedError("HunyuanVideo 1.5 on the neuron backend runs at TP1 (the whole "
                                      "DiT on every core), with context parallelism for long videos")
        self.cp = int(getattr(self.parallel, "cp_degree", 1) or 1)
        if self.cp > 1 and getattr(self.parallel, "cp_mode", "ulysses") != "ulysses":
            raise NotImplementedError("HunyuanVideo 1.5 runs context parallelism in Ulysses mode only")
        with open(os.path.join(str(self.model_path), "config.json"), encoding="utf-8") as f:
            self.model_config = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
        if int(num_frames) % VAE_TEMPORAL != 1:
            raise ValueError(f"num_frames={num_frames} must be 4k+1")
        for name, value in (("height", height), ("width", width)):
            if int(value) % VAE_SPATIAL:
                raise ValueError(f"{name}={value} must be a multiple of {VAE_SPATIAL}")
        self.latent_shape = (int(batch_size), int(self.model_config["in_channels"]),
                             (int(num_frames) - 1) // VAE_TEMPORAL + 1,
                             int(height) // VAE_SPATIAL, int(width) // VAE_SPATIAL)
        self.text_seq_len, self.text_2_seq_len = int(text_seq_len), int(text_2_seq_len)
        self.image_tokens, self.batch_size = int(image_tokens), int(batch_size)
        _, _, f, h, w = self.latent_shape
        self.num_image_tokens = f * h * w  # patch size 1 x 1 x 1
        if self.cp > 1:
            heads = int(self.model_config["num_attention_heads"])
            if heads % self.cp:
                raise ValueError(f"Ulysses needs num_attention_heads={heads} divisible by cp={self.cp}")
            if self.num_image_tokens % self.cp:
                raise ValueError(
                    f"{self.num_image_tokens} image tokens ({f} x {h} x {w}) do not divide by cp={self.cp}; "
                    f"pick a frame count or CP degree that divides them (e.g. 121 frames at 480x848 is "
                    f"31 x 30 x 53: cp=2)")

    def build_module(self) -> torch.nn.Module:
        model = _tensor_signature_model()(**self.model_config)
        if self.cp > 1:
            _install_context_parallel(model, self.num_image_tokens, self.cp)
        return model

    def checkpoint_key(self, name: str) -> str:
        return name  # diffusers' layout, key for key

    def _text_to_video_defaults(self, encoder_hidden_states_2, encoder_attention_mask_2, image_embeds):
        """What the pipeline passes for text-to-video without quoted text: an empty glyph stream and
        zero image embeddings."""
        b, dt = self.batch_size, self.dtype
        if encoder_hidden_states_2 is None:
            encoder_hidden_states_2 = torch.zeros(b, self.text_2_seq_len,
                                                  int(self.model_config["text_embed_2_dim"]), dtype=dt)
            encoder_attention_mask_2 = torch.zeros(b, self.text_2_seq_len, dtype=torch.int64)
        if image_embeds is None:
            image_embeds = torch.zeros(b, self.image_tokens, int(self.model_config["image_embed_dim"]), dtype=dt)
        return encoder_hidden_states_2, encoder_attention_mask_2, image_embeds

    def get_example_inputs(self) -> tuple[torch.Tensor, ...]:
        b, dt = self.batch_size, self.dtype
        mask = torch.zeros(b, self.text_seq_len, dtype=torch.int64)
        mask[:, :64] = 1
        e2, m2, img = self._text_to_video_defaults(None, None, None)
        return self._positional(torch.zeros(self.latent_shape, dtype=dt), torch.ones(b, dtype=dt),
                                torch.zeros(b, self.text_seq_len, int(self.model_config["text_embed_dim"]), dtype=dt),
                                mask, e2, m2, img)

    def _positional(self, *tensors) -> tuple[torch.Tensor, ...]:
        """The module's seven tensors on this rank's device: floats in the application dtype, the
        masks (integer) as they are."""
        dev, dt = self.device, self.dtype
        return tuple(t.to(dev, dt) if t.is_floating_point() else t.to(dev) for t in tensors)

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                timestep_r=None, encoder_hidden_states_2=None, encoder_attention_mask_2=None,
                image_embeds=None, attention_kwargs=None, return_dict=False):
        """The pipeline's call; returns ``(noise_pred,)`` on the device."""
        if timestep_r is not None:
            raise NotImplementedError("timestep_r (meanflow) is not supported")
        e2, m2, img = self._text_to_video_defaults(encoder_hidden_states_2, encoder_attention_mask_2,
                                                   image_embeds)
        return (super().forward(*self._positional(hidden_states, timestep, encoder_hidden_states,
                                                  encoder_attention_mask, e2, m2, img)),)


class _JointUlyssesProcessor:
    """``HunyuanVideo15AttnProcessor2_0`` with the attention done by ``joint_ulysses_attention``:
    the image stream arrives sharded (this rank's ``N/cp`` tokens, rotary rows to match), the text
    stream replicated; projections, QK norms and rotary are diffusers' own. Keys are masked by
    count (valid image keys, then the valid text keys diffusers reordered to the front)."""

    def __init__(self, num_image_tokens: int) -> None:
        self.num_image_tokens = int(num_image_tokens)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None):
        from diffusers.models.embeddings import apply_rotary_emb

        from difflet.backends.neuron.ops_impl.attention import joint_ulysses_attention

        def heads(t):
            return t.unflatten(2, (attn.heads, -1))

        q, k, v = heads(attn.to_q(hidden_states)), heads(attn.to_k(hidden_states)), heads(attn.to_v(hidden_states))
        q, k = attn.norm_q(q), attn.norm_k(k)
        if image_rotary_emb is not None:
            q = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
            k = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)
        eq = heads(attn.add_q_proj(encoder_hidden_states))
        ek = heads(attn.add_k_proj(encoder_hidden_states))
        ev = heads(attn.add_v_proj(encoder_hidden_states))
        if attn.norm_added_q is not None:
            eq = attn.norm_added_q(eq)
        if attn.norm_added_k is not None:
            ek = attn.norm_added_k(ek)
        valid_text = attention_mask.to(torch.int32).sum(dim=1)
        key_valid_len = (valid_text + self.num_image_tokens).to(torch.int32)
        bhsd = lambda t: t.transpose(1, 2).contiguous()  # noqa: E731  [B, S, H, d] -> [B, H, S, d]
        img_out, txt_out = joint_ulysses_attention(
            bhsd(q), bhsd(eq), bhsd(k), bhsd(v), bhsd(ek), bhsd(ev),
            scale=q.shape[-1] ** -0.5, key_valid_len=key_valid_len)
        img_out = img_out.transpose(1, 2).flatten(2, 3).to(q.dtype)
        txt_out = txt_out.transpose(1, 2).flatten(2, 3).to(q.dtype)
        img_out = attn.to_out[1](attn.to_out[0](img_out))
        txt_out = attn.to_add_out(txt_out)
        return img_out, txt_out


def _install_context_parallel(model, num_image_tokens: int, cp: int) -> None:
    """Image tokens split over the CP group around the blocks: before each block, this rank's
    ``N/cp`` image rows (block 0 receives all ``N``) and the matching rotary rows; after the last
    block, an all-gather over the CP group. Plus the joint-Ulysses processor in every block."""
    from difflet.backends.neuron.ops_impl import collectives
    from difflet.backends.neuron.ops_impl.parallel_mesh import get_axis_rank

    n, local = int(num_image_tokens), int(num_image_tokens) // cp
    start = get_axis_rank("cp") * local

    def shard(t):
        if not isinstance(t, torch.Tensor):
            return t
        for dim, size in enumerate(t.shape):
            if size == n:
                return t.narrow(dim, start, local)
        return t

    def before(block, args):
        hidden_states, encoder_hidden_states, temb, mask, freqs = args
        if hidden_states.shape[1] == n:
            hidden_states = hidden_states.narrow(1, start, local).contiguous()
        if isinstance(freqs, (tuple, list)):
            freqs = type(freqs)(shard(f) for f in freqs)
        else:
            freqs = shard(freqs)
        return hidden_states, encoder_hidden_states, temb, mask, freqs

    def after(block, args, output):
        hidden_states, encoder_hidden_states = output
        gathered = collectives.gather_from_tensor_model_parallel_region_with_dim(
            hidden_states.contiguous(), 1, collectives.get_cp_group())
        return gathered, encoder_hidden_states

    processor = _JointUlyssesProcessor(n)
    for block in model.transformer_blocks:
        block.attn.set_processor(processor)
        block.register_forward_pre_hook(before)
    model.transformer_blocks[-1].register_forward_hook(after)


def _tensor_signature_model():
    """diffusers' model with a tensor-only forward (the application base warms up and compiles
    with tensor inputs): ``(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
    encoder_hidden_states_2, encoder_attention_mask_2, image_embeds) -> noise_pred``. A subclass,
    so parameter names and ``transformer_blocks`` are diffusers' own."""
    from diffusers import HunyuanVideo15Transformer3DModel

    class HunyuanVideo15TensorSignature(HunyuanVideo15Transformer3DModel):
        def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                    encoder_hidden_states_2, encoder_attention_mask_2, image_embeds):
            return super().forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                                   encoder_hidden_states_2=encoder_hidden_states_2,
                                   encoder_attention_mask_2=encoder_attention_mask_2,
                                   image_embeds=image_embeds, return_dict=False)[0]

    return HunyuanVideo15TensorSignature


__all__ = ["NeuronHunyuanVideo15TransformerApplication", "VAE_SPATIAL", "VAE_TEMPORAL"]
