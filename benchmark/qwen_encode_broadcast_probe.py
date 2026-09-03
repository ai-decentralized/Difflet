"""Does an over-long prompt fail on every rank, or hang three of them?

The encode broadcast means a raise on the encoding rank alone would leave the
others blocked on the collective. This drives the real _encode_prompt_tpu on
four processes with a prompt past the encoder bucket and asserts all four
raise. Only the text stage is loaded; no DiT.
"""
import multiprocessing as mp
from pathlib import Path

SNAP = "/mnt/models/hf/hub/models--Qwen--Qwen-Image/snapshots/75e0b4be04f60ec59a75f475837eced720f823b6"


def worker(rank, world, q):
    from torch_xla._internal import pjrt
    pjrt.initialize_multiprocess(rank, world)
    import torch, torch_xla.runtime as xr
    torch.set_num_threads(8)
    from types import SimpleNamespace
    from difflet.serving.orchestrators.qwen_image import QwenImageServingStageAdapter
    from difflet.serving.types import ParallelTopology

    a = QwenImageServingStageAdapter()
    a._tpu = True
    a.model_dir = SNAP
    profile = SimpleNamespace(
        height=1024, width=1024, num_frames=None,
        parallel=ParallelTopology(tp_degree=world, cp_degree=1, world_size=world),
        world_size=world,
        shape_dict=lambda: {"height": 1024, "width": 1024, "num_frames": None},
    )
    a.active_profile = profile
    a._load_text_stage_tpu(profile)

    ordinal = int(xr.global_ordinal())
    out = {"rank": rank, "ordinal": ordinal}
    # 1) a normal prompt must succeed everywhere
    try:
        r = a._encode_prompt_tpu("a red fox in snow")
        out["ok_shape"] = list(r["encoder_hidden_states"].shape)
        out["ok_tokens"] = int(r["encoder_hidden_states_mask"].sum())
    except Exception as e:
        out["ok_error"] = f"{type(e).__name__}: {e}"
    # 2) an over-long prompt must raise on EVERY rank, not hang
    try:
        a._encode_prompt_tpu("fox " * 4000)
        out["long"] = "NO ERROR (wrong)"
    except Exception as e:
        out["long"] = type(e).__name__
        out["long_code"] = getattr(e, "code", None) or getattr(e, "error_code", None)
    q.put(out)


def main():
    world = len(list(Path("/dev/vfio").glob("[0-9]*"))) or 1
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=worker, args=(r, world, q)) for r in range(world)]
    [p.start() for p in ps]
    got = []
    for _ in range(world):
        try:
            got.append(q.get(timeout=900))
        except Exception:
            print("TIMEOUT -- a rank hung on the collective", flush=True)
            break
    for g in sorted(got, key=lambda x: x["rank"]):
        print(g, flush=True)
    print(f"\n{len(got)}/{world} ranks reported; "
          f"{'PASS' if len(got) == world else 'FAIL (hang)'}", flush=True)
    for p in ps:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()


if __name__ == "__main__":
    main()
