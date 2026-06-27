#!/usr/bin/env python
"""Task 3 on-device parity spike for ``joint_ring_attention`` (tp=2/cp=2).

Compiles a tiny SPMD probe with NxD ModelBuilder. The probe receives the FULL
(replicated) synthetic joint tensors on every rank and, using the cp/data-parallel
group, scatters the image stream into per-rank shards (``S_img/cp``) while keeping
the text stream replicated (``S_txt``). It then computes, in one graph:

  * candidate = ``joint_ring_attention`` over the per-rank image shard + text
    (a genuine ``collective_permute`` ring across the cp group), and
  * reference = ONE full joint ``attention`` over the gathered ``[image ‖ text]``.

Both use this rank's joint query ``S_q = S_img/cp + S_txt``. The two are stacked
and returned; the host asserts cosine >= 0.999 (ring is lossless vs gather-KV).
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from difflet.backends.trainium.core.application_base import NeuronApplicationBase
from difflet.backends.trainium.core.config import InferenceConfig, NeuronConfig
from difflet.backends.trainium.core.model_wrapper import BaseModelInstance, ModelWrapper


# ---------------------------------------------------------------------------
# Probe module: candidate (joint ring) vs reference (gather-KV joint attention)


class JointRingSpikeProbe(nn.Module):
    def __init__(self, config: InferenceConfig):
        super().__init__()
        from difflet.backends.trainium.ops_impl.collectives import (
            SPMDRank,
            get_data_parallel_group,
            get_world_group,
        )

        self.scale = float(config.spike_scale)
        self.data_parallel_group = get_data_parallel_group()
        self.global_rank = SPMDRank(world_size=get_world_group().size())

    def forward(self, image_q, text_q, image_k, image_v, text_k, text_v):
        from difflet.backends.trainium.ops_impl.collectives import (
            get_dp_rank_spmd,
            get_tensor_model_parallel_size,
            scatter_to_process_group_spmd,
        )
        from difflet.ops import attention, joint_ring_attention

        dp_rank = get_dp_rank_spmd(
            global_rank=self.global_rank.get_rank(),
            tp_degree=get_tensor_model_parallel_size(),
        )

        # Image stream is sharded across the cp ring; text stream is replicated.
        iq = scatter_to_process_group_spmd(
            image_q, partition_dim=2, rank=dp_rank, process_group=self.data_parallel_group
        )
        ik = scatter_to_process_group_spmd(
            image_k, partition_dim=2, rank=dp_rank, process_group=self.data_parallel_group
        )
        iv = scatter_to_process_group_spmd(
            image_v, partition_dim=2, rank=dp_rank, process_group=self.data_parallel_group
        )

        # This rank's joint query: its image-shard queries + the replicated text.
        q = torch.cat([iq, text_q], dim=2)

        candidate = joint_ring_attention(
            q, ik, iv, text_k, text_v, scale=self.scale, causal=False
        )

        # Reference: full joint attention over the gathered [image ‖ text] keys.
        b, h, s_q, d = q.shape
        full_k = torch.cat([image_k, text_k], dim=2)
        full_v = torch.cat([image_v, text_v], dim=2)
        s_k = full_k.shape[2]
        reference = attention(
            q.reshape(b * h, s_q, d),
            full_k.reshape(b * h, s_k, d),
            full_v.reshape(b * h, s_k, d),
            scale=self.scale, causal=False, tp_q=True, tp_k=True, tp_out=False,
        ).reshape(b, h, s_q, d)

        return torch.stack([candidate, reference], dim=0)


# ---------------------------------------------------------------------------
# Minimal NxD config / wrapper / application


class JointRingSpikeConfig(InferenceConfig):
    def get_required_attributes(self):
        return []

    def add_derived_config(self):
        super().add_derived_config()
        self.pad_token_id = 0


class ModelWrapperJointRingSpike(ModelWrapper):
    def __init__(self, config, model_cls, tag="", compiler_args=None, priority_model_idx=None):
        super().__init__(config, model_cls, tag, compiler_args, priority_model_idx)
        self.bucket_config = None

    def input_generator(self):
        c = self.config
        dt = c.neuron_config.torch_dtype
        full = lambda s: torch.randn([c.spike_b, c.spike_h, s, c.spike_d], dtype=dt)
        return [(
            full(c.spike_s_img),  # image_q
            full(c.spike_s_txt),  # text_q
            full(c.spike_s_img),  # image_k
            full(c.spike_s_img),  # image_v
            full(c.spike_s_txt),  # text_k
            full(c.spike_s_txt),  # text_v
        )]

    def get_model_instance(self):
        def _create_model():
            model = self.model_cls(self.config).to(dtype=self.config.neuron_config.torch_dtype)
            model.eval()
            return model

        return BaseModelInstance(module_cls=_create_model, input_output_aliases={})

    def forward(self, *args):
        if self.model is None:
            raise RuntimeError("Forward called before load. Run load() first.")
        return self._forward(*args)


class JointRingSpikeApplication(NeuronApplicationBase):
    _model_cls = JointRingSpikeProbe

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = ModelWrapperJointRingSpike(
            config=self.config,
            model_cls=self._model_cls,
            tag=self._model_cls.__name__,
            compiler_args=self.get_compiler_args(),
            priority_model_idx=0,
        )
        self.models.append(self.model)

    @classmethod
    def get_config_cls(cls):
        return JointRingSpikeConfig

    def forward(self, *model_inputs):
        return self.models[0](*model_inputs)

    def get_compiler_args(self) -> str:
        os.environ["LOCAL_WORLD_SIZE"] = str(self.config.neuron_config.world_size)
        return (
            "--model-type=transformer -O1 "
            "--tensorizer-options='--enable-ccop-compute-overlap' "
            "--auto-cast=none "
            "--internal-hlo2tensorizer-options='--verify-hlo=true'"
        )

    def checkpoint_loader_fn(self, mmap: bool = False):
        # Parameter-free probe except SPMDRank's `rank` buffer, which must hold
        # arange(world_size) so ModelBuilder shards each rank its own id.
        world_size = self.config.neuron_config.world_size
        return {"global_rank.rank": torch.arange(world_size, dtype=torch.int32)}


def main() -> int:
    tp_degree = int(os.environ.get("DIFFLET_JOINT_RING_TP_DEGREE", "2"))
    cp_degree = int(os.environ.get("DIFFLET_JOINT_RING_CP_DEGREE", "2"))
    world_size = tp_degree * cp_degree
    cosine_min = float(os.environ.get("DIFFLET_JOINT_RING_COSINE_MIN", "0.999"))
    workdir = os.environ.get("DIFFLET_JOINT_RING_WORKDIR", "/tmp/joint_ring_spike")

    b = int(os.environ.get("DIFFLET_JOINT_RING_B", "1"))
    h = int(os.environ.get("DIFFLET_JOINT_RING_H", "8"))
    s_img = int(os.environ.get("DIFFLET_JOINT_RING_S_IMG", "256"))   # total image seq
    s_txt = int(os.environ.get("DIFFLET_JOINT_RING_S_TXT", "128"))   # replicated text seq
    d = int(os.environ.get("DIFFLET_JOINT_RING_D", "128"))
    seed = int(os.environ.get("DIFFLET_JOINT_RING_SEED", "1234"))

    if s_img % cp_degree != 0 or (s_img // cp_degree) % 128 != 0:
        raise ValueError(
            f"S_img ({s_img}) must be divisible by cp ({cp_degree}) and the per-rank "
            f"shard S_img/cp must be a multiple of 128 (got {s_img // cp_degree})"
        )

    scale = 1.0 / (d ** 0.5)
    dtype = torch.bfloat16

    neuron_config = NeuronConfig(
        batch_size=b,
        tp_degree=tp_degree,
        cp_degree=cp_degree,
        world_size=world_size,
        torch_dtype=dtype,
        skip_sharding=True,
    )
    config = JointRingSpikeConfig(
        neuron_config=neuron_config,
        spike_b=b, spike_h=h, spike_s_img=s_img, spike_s_txt=s_txt,
        spike_d=d, spike_scale=scale,
    )

    per_rank = s_img // cp_degree
    s_q = per_rank + s_txt
    print(f"[joint-parity] tp={tp_degree} cp={cp_degree} world={world_size} "
          f"B={b} H={h} S_img={s_img} (per_rank={per_rank}) S_txt={s_txt} d={d} S_q={s_q}")

    out_dir = os.path.join(workdir, "compile")
    os.makedirs(out_dir, exist_ok=True)

    model_path = tempfile.mkdtemp(prefix="joint_ring_probe_")
    app = JointRingSpikeApplication(model_path=model_path, config=config)
    print(f"[joint-parity] compiling -> {out_dir}")
    app.compile(out_dir)
    print("[joint-parity] loading weights to device")
    app.load(out_dir)

    # Fixed-seed FULL (replicated) inputs; each rank scatters its own image shard.
    torch.manual_seed(seed)
    full = lambda s: torch.randn([b, h, s, d], dtype=dtype)
    inputs = (full(s_img), full(s_txt), full(s_img), full(s_img), full(s_txt), full(s_txt))

    print("[joint-parity] running forward")
    with torch.no_grad():
        out = app.forward(*inputs)
    out = out.detach().to(torch.float32).cpu()
    candidate, reference = out[0], out[1]

    diff = (candidate - reference).abs()
    cos = float(F.cosine_similarity(candidate.flatten(), reference.flatten(), dim=0))
    metrics = {
        "shape": list(candidate.shape),
        "cosine": cos,
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rmse": float(torch.sqrt((diff * diff).mean())),
        "cosine_min": cosine_min,
    }
    print("[joint-parity] metrics:", metrics)
    if cos < cosine_min:
        print(f"[joint-parity] FAIL — joint ring vs gather_kv cosine {cos:.6f} < {cosine_min}: {metrics}")
        return 1
    print(f"[joint-parity] PASS — joint ring matches gather_kv (cosine {cos:.6f} >= {cosine_min})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
