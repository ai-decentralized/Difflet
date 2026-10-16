"""Text-embedding parity on the host CPU: the neuron Qwen-Image application's prompt encode
(``_HostQwenTextEncoder._encode``: the Trainium text stage's template, drop index 34, 256-token
budget, valid prefix only, zero-padded to TEXT_SEQ_LEN in bf16) against diffusers'
``QwenImagePipeline._get_qwen_prompt_embeds`` for the same prompt, on the valid rows.

No device: both encoders run on the CPU. The application's encoder runs in bf16 (its default);
diffusers in bf16 and, with ``--fp32``, in fp32 as well (the precision floor of the bf16 pair).
Optionally ``--device-embeds`` is a ``{"embeds", "mask"}`` .pt dumped by a CLI rank 0 (the V3
probe) and is compared too, to show the CLI broadcast exactly what the host encode computes.

    PYTHONPATH=<worktree> <venv>/bin/python tests/manual/check_qwen_image_prompt_embeds.py \
        --model-dir <snapshot root> --prompt "..." [--fp32] [--device-embeds x.pt] [--out-json r.json]
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.double().flatten(), b.double().flatten()
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "rel_l1": float(diff.sum() / b.abs().sum()),
        "cosine": float(torch.nn.functional.cosine_similarity(a, b, dim=0)),
        "equal": bool(torch.equal(a, b)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--device-embeds", default=None)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    from diffusers import QwenImagePipeline
    from transformers import AutoTokenizer, Qwen2_5_VLForConditionalGeneration

    from difflet.common.orchestrators.qwen_image import QWEN_TEMPLATE, TEXT_SEQ_LEN
    from difflet.models.qwen_image.neuron_application import _HostQwenTextEncoder

    result: dict = {"prompt": args.prompt}
    tok = AutoTokenizer.from_pretrained(f"{args.model_dir}/tokenizer")
    t = time.perf_counter()
    te = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        f"{args.model_dir}/text_encoder", dtype=torch.bfloat16).eval()
    result["load_bf16_s"] = round(time.perf_counter() - t, 2)

    enc = _HostQwenTextEncoder(args.model_dir, text_dim=3584, seq_len=TEXT_SEQ_LEN,
                               dtype=torch.bfloat16, device="cpu", threads=args.threads)
    enc.tokenizer, enc.model = tok, te
    t = time.perf_counter()
    ours, valid = enc._encode(args.prompt)
    result["ours_encode_s"] = round(time.perf_counter() - t, 3)
    result["ours_valid_rows"] = valid
    result["ours_padding_zero"] = bool((ours[:, valid:] == 0).all())

    pipe = QwenImagePipeline(scheduler=None, vae=None, text_encoder=te, tokenizer=tok,
                             transformer=None)
    result["template_equal"] = pipe.prompt_template_encode == QWEN_TEMPLATE
    with torch.no_grad():
        t = time.perf_counter()
        ref, ref_mask = pipe._get_qwen_prompt_embeds(args.prompt, device=torch.device("cpu"))
        result["diffusers_bf16_encode_s"] = round(time.perf_counter() - t, 3)
    result["diffusers_rows"] = int(ref_mask.sum())
    n = min(valid, ref.shape[1])
    result["ours_vs_diffusers_bf16"] = metrics(ours[:, :n].float(), ref[:, :n].float())

    if args.device_embeds:
        dev = torch.load(args.device_embeds, map_location="cpu")
        result["device_rows"] = int(dev["mask"].sum())
        result["device_vs_ours"] = metrics(dev["embeds"].float(), ours.float())
        result["device_vs_diffusers_bf16"] = metrics(dev["embeds"][:, :n].float(),
                                                     ref[:, :n].float())
    if args.fp32:
        del pipe
        te32 = te.to(torch.float32)
        pipe = QwenImagePipeline(scheduler=None, vae=None, text_encoder=te32, tokenizer=tok,
                                 transformer=None)
        with torch.no_grad():
            ref32, _ = pipe._get_qwen_prompt_embeds(args.prompt, device=torch.device("cpu"))
        result["ours_vs_diffusers_fp32"] = metrics(ours[:, :n].float(), ref32[:, :n].float())
        result["diffusers_bf16_vs_fp32"] = metrics(ref[:, :n].float(), ref32[:, :n].float())
        result["note_fp32"] = "fp32 = the bf16 checkpoint weights upcast, so it isolates compute precision"
    print(json.dumps(result, indent=2))
    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
