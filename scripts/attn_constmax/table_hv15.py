"""HunyuanVideo-1.5's const-max table: ``table.py``'s formulas and gap rule, unchanged, on the
snapshot's ``transformer_blocks.{i}.attn`` QK-RMSNorm gains (norm_q / norm_k on the image stream,
norm_added_q / norm_added_k on the text: one key-bounded call covers all four pairs).

Reference Sk = 51,275, the call's joint key length at 480x848x121: 31 x 30 x 53 = 49,290 image
tokens + 1,985 text (1,000 Qwen2.5-VL + 256 ByT5 + 729 image-embedding tokens). The backend
recomputes softmax_max at each call's own Sk, so this only sets ``gap``.

usage: table_hv15.py OUT_JSON
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import table  # noqa: E402

table.MODELS["hunyuan_video_15"] = (
    "HunyuanVideo-1.5 (joint, 480x848x121)",
    "models--hunyuanvideo-community--HunyuanVideo-1.5-Diffusers-480p_t2v",
    r"transformer_blocks\.\d+\.attn\.", 31 * 30 * 53 + 1000 + 256 + 729)

if __name__ == "__main__":
    table.main("hunyuan_video_15", Path(sys.argv[1]), None)
