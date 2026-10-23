"""Neuron application for HunyuanVideo 1.5's DiT: diffusers' ``HunyuanVideo15Transformer3DModel``.

The module is diffusers' own (no Difflet port exists; the Trainium path wraps the same class), built
on ``meta`` and loaded from diffusers' checkpoint key for key, fp32 -> the application dtype; in
compile mode its 54 identical ``transformer_blocks`` are compiled in place into one graph.

diffusers' forward runs as is: its text-token reorder (valid tokens of the image, ByT5 and
Qwen2.5-VL streams first, padding zeroed after) is data-dependent but fixed-length (729 + 256 + 1000
= 1,985 tokens), and it runs in the un-compiled top-level forward, so the compiled blocks see one
shape. (The Trainium path forks it because it compiles the whole model as one graph.)

Parallelism. Tensor parallelism replaces diffusers' linears with the backend's column/row-parallel
layers under the same names (Megatron layout: q/k/v and the added text q/k/v column-parallel, the
attention outputs row-parallel, each MLP column then row, the modulation linears column-parallel
with the output gathered; heads split 16/tp; the token refiner likewise), so the per-rank loader
shards diffusers' checkpoint unchanged. Context parallelism (Ulysses) splits the image tokens across
the CP group around the blocks, the text stream (729 + 256 + 1000 tokens) replicated.

CFG parallelism (the checkpoint's guider runs classifier-free guidance, scale 6.0, on every step):
with ``cfg_parallel_enabled`` the cores form two groups along the mesh's cfg axis; ``forward`` takes
the two branches stacked as a batch of 2, runs this group's branch (a batch of 1: the compiled
shapes are unchanged), and all-gathers both predictions over the cfg group, as Wan's CFG mode does.

Attention. The blocks' processor is replaced at every layout: diffusers' builds a dense symmetric
(N+T)^2 mask (about 2.6 GB per block at 121 frames) for masked SDPA. Ours expresses the text padding
as a key count, N image keys + the valid text keys (diffusers reorders valid text tokens first),
which on the device runs attention_cte's key-bounded flash kernel in O(L) memory, through
``joint_ulysses_attention`` under CP. It is exact for every valid query: the symmetric and key masks
differ only on padded text queries, whose outputs are never used.
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
        self.tp = int(self.parallel.tp_degree)
        self.cp = int(getattr(self.parallel, "cp_degree", 1) or 1)
        self.cfg = 2 if bool(getattr(self.parallel, "cfg_parallel_enabled", False)) else 1
        if self.cfg > 1 and int(batch_size) != 1:
            raise ValueError("CFG parallelism runs one branch per group: batch_size must be 1")
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
        heads = int(self.model_config["num_attention_heads"])
        if heads % self.tp:
            raise ValueError(f"num_attention_heads={heads} must divide by tp={self.tp}")
        if self.cp > 1:
            if (heads // self.tp) % self.cp:
                raise ValueError(f"Ulysses needs the {heads // self.tp} heads per TP rank divisible "
                                 f"by cp={self.cp}")
            if self.num_image_tokens % self.cp:
                raise ValueError(
                    f"{self.num_image_tokens} image tokens ({f} x {h} x {w}) do not divide by cp={self.cp}; "
                    f"pick a frame count or CP degree that divides them (e.g. 121 frames at 480x848 is "
                    f"31 x 30 x 53: cp=2)")

    def build_module(self) -> torch.nn.Module:
        model = _tensor_signature_model()(**self.model_config)
        if self.tp > 1:
            _install_tensor_parallel(model, self.tp)
        if self.cp > 1:
            _install_context_parallel(model, self.num_image_tokens, self.cp)
        else:
            processor = _JointBoundedProcessor()
            for block in model.transformer_blocks:
                block.attn.set_processor(processor)
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

    @torch.no_grad()
    def teacache_mod_input(self, hidden_states, timestep) -> torch.Tensor:
        """The TeaCache signal as the Trainium 1.5 probe defines it: block 0's modulated image input,
        ``transformer_blocks[0].norm1(x_embedder(x), time_embed(t))[0]``. The modulation is
        timestep-only (text and image conditioning enter as attention tokens), so the signal does
        not depend on the guidance branch; under TP the modulation linear gathers its output, so
        every rank holds it whole. Host inputs; moved to this rank's device."""
        model = self.module
        x = hidden_states.to(self.device, self.dtype)
        t = timestep.to(self.device, self.dtype)
        temb = model.time_embed(t, timestep_r=None)
        modulated, *_ = model.transformer_blocks[0].norm1(model.x_embedder(x), emb=temb)
        return modulated

    def forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                timestep_r=None, encoder_hidden_states_2=None, encoder_attention_mask_2=None,
                image_embeds=None, attention_kwargs=None, return_dict=False):
        """The pipeline's call; returns ``(noise_pred,)`` on the device."""
        if timestep_r is not None:
            raise NotImplementedError("timestep_r (meanflow) is not supported")
        if self.cfg > 1:
            return (self._cfg_forward(hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                                      encoder_hidden_states_2, encoder_attention_mask_2, image_embeds),)
        e2, m2, img = self._text_to_video_defaults(encoder_hidden_states_2, encoder_attention_mask_2,
                                                   image_embeds)
        return (super().forward(*self._positional(hidden_states, timestep, encoder_hidden_states,
                                                  encoder_attention_mask, e2, m2, img)),)

    def _cfg_forward(self, hidden_states, timestep, encoder_hidden_states, encoder_attention_mask,
                     encoder_hidden_states_2, encoder_attention_mask_2, image_embeds):
        """Inputs stacked [uncond, cond] (batch 2; a batch-1 tensor is shared by both branches): this
        group's branch through the DiT, then both predictions all-gathered over the cfg group."""
        from difflet.backends.neuron.ops_impl import collectives
        from difflet.backends.neuron.ops_impl.parallel_mesh import get_axis_rank

        branch = get_axis_rank("cfg")

        def pick(t):
            if t is None:
                return None
            if t.shape[0] == 2:
                return t[branch:branch + 1]
            if t.shape[0] == 1:
                return t
            raise ValueError(f"CFG parallelism expects inputs of batch 2 (uncond, cond), got {tuple(t.shape)}")

        e2, m2, img = self._text_to_video_defaults(pick(encoder_hidden_states_2), pick(encoder_attention_mask_2),
                                                   pick(image_embeds))
        out = super().forward(*self._positional(pick(hidden_states), pick(timestep), pick(encoder_hidden_states),
                                                pick(encoder_attention_mask), e2, m2, img))
        return collectives.gather_from_tensor_model_parallel_region_with_dim(
            out.contiguous(), 0, collectives.get_cfg_group())


def _joint_qkv(attn, hidden_states, encoder_hidden_states, image_rotary_emb):
    """diffusers' HunyuanVideo15AttnProcessor2_0 up to the attention: projections, QK norms, rotary
    on the image stream; ``[B, H, S, d]`` (this rank's heads) for image and text."""
    from diffusers.models.embeddings import apply_rotary_emb

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
    bhsd = lambda t: t.transpose(1, 2).contiguous()  # noqa: E731  [B, S, H, d] -> [B, H, S, d]
    return tuple(bhsd(t) for t in (q, k, v, eq, ek, ev))


def _joint_out(attn, img_out, txt_out, dtype):
    """``[B, H, S, d]`` attention outputs through diffusers' output projections."""
    img_out = img_out.transpose(1, 2).flatten(2, 3).to(dtype)
    txt_out = txt_out.transpose(1, 2).flatten(2, 3).to(dtype)
    return attn.to_out[1](attn.to_out[0](img_out)), attn.to_add_out(txt_out)


class _JointBoundedProcessor:
    """``HunyuanVideo15AttnProcessor2_0`` with the text padding as a key count instead of a dense
    symmetric mask: queries attend to keys ``[0, N + valid text)`` of the joint ``[image || text]``
    sequence (attention_cte's key-bounded flash kernel on the device)."""

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None):
        from difflet.backends.neuron.ops_impl.attention import attention

        q, k, v, eq, ek, ev = _joint_qkv(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        n = q.shape[2]
        b, h, s = q.shape[0], q.shape[1], n + eq.shape[2]
        q, k, v = (torch.cat(pair, dim=2).reshape(b * h, s, -1) for pair in ((q, eq), (k, ek), (v, ev)))
        valid = attention_mask.to(torch.int32).sum(dim=1) + n  # [B]
        hi = valid.view(b, 1, 1).expand(b, h, s).reshape(b * h, s, 1).to(torch.int32)
        out = attention(q, k, v, scale=q.shape[-1] ** -0.5, causal=False,
                        bound_min=torch.zeros_like(hi), bound_max=hi, tp_q=True, tp_k=True, tp_out=False)
        out = out.reshape(b, h, s, -1)
        return _joint_out(attn, out[:, :, :n], out[:, :, n:], hidden_states.dtype)


class _JointUlyssesProcessor:
    """``HunyuanVideo15AttnProcessor2_0`` with the attention done by ``joint_ulysses_attention``:
    the image stream arrives sharded (this rank's ``N/cp`` tokens, rotary rows to match), the text
    stream replicated; projections, QK norms and rotary are diffusers' own. Keys are masked by
    count (valid image keys, then the valid text keys diffusers reordered to the front)."""

    def __init__(self, num_image_tokens: int) -> None:
        self.num_image_tokens = int(num_image_tokens)

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None,
                 image_rotary_emb=None):
        from difflet.backends.neuron.ops_impl.attention import joint_ulysses_attention

        q, k, v, eq, ek, ev = _joint_qkv(attn, hidden_states, encoder_hidden_states, image_rotary_emb)
        valid_text = attention_mask.to(torch.int32).sum(dim=1)
        key_valid_len = (valid_text + self.num_image_tokens).to(torch.int32)
        img_out, txt_out = joint_ulysses_attention(q, eq, k, v, ek, ev, scale=q.shape[-1] ** -0.5,
                                                   key_valid_len=key_valid_len)
        return _joint_out(attn, img_out, txt_out, hidden_states.dtype)


def _install_tensor_parallel(model, tp: int) -> None:
    """diffusers' linears -> the backend's column/row-parallel layers, same attribute names (so the
    loader's ``_difflet_shard`` declarations shard the checkpoint): Megatron layout per block and per
    token-refiner block; modulation linears column-parallel with the output gathered; heads / tp."""
    from difflet.backends.neuron.ops_impl.linear import ColumnParallelLinear, RowParallelLinear

    def col(lin, gather=False):
        return ColumnParallelLinear(lin.in_features, lin.out_features, bias=lin.bias is not None,
                                    gather_output=gather, dtype=lin.weight.dtype, device=lin.weight.device)

    def row(lin):
        return RowParallelLinear(lin.in_features, lin.out_features, bias=lin.bias is not None,
                                 input_is_parallel=True, dtype=lin.weight.dtype, device=lin.weight.device)

    def attention(attn, joint):
        names = ("to_q", "to_k", "to_v") + (("add_q_proj", "add_k_proj", "add_v_proj") if joint else ())
        for name in names:
            setattr(attn, name, col(getattr(attn, name)))
        attn.to_out[0] = row(attn.to_out[0])
        if joint:
            attn.to_add_out = row(attn.to_add_out)
        attn.heads //= tp

    def mlp(ff):
        ff.net[0].proj = col(ff.net[0].proj)
        ff.net[2] = row(ff.net[2])

    for block in model.transformer_blocks:
        attention(block.attn, joint=True)
        mlp(block.ff)
        mlp(block.ff_context)
        block.norm1.linear = col(block.norm1.linear, gather=True)
        block.norm1_context.linear = col(block.norm1_context.linear, gather=True)
    for block in model.context_embedder.token_refiner.refiner_blocks:
        attention(block.attn, joint=False)
        mlp(block.ff)
        block.norm_out.linear = col(block.norm_out.linear, gather=True)


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
