"""Experiment E1, lane HV1: the const-max table for HunyuanVideo 1.0, by ``table.py``'s code and rule
(sigma, softmax_max = 0.88 sigma sqrt(2 ln Sk), hard_bound, gap <= 80) with HV1's entry added here
rather than in ``table.py`` (other lanes edit it). HV1's joint attention is FLUX's layout: 20
dual-stream blocks (image q/k through norm_q/norm_k, text through norm_added_q/norm_added_k, so four
(q norm, k norm) pairs per call) and 40 single-stream blocks (one pair), the same module regex.
The token refiner's self-attention has no QK norm and a dense mask: it is not in the table.

Reference Sk: the joint key length at 544x960x129 (``hunyuan_video_L``): 33 x 34 x 60 image tokens
+ 256 right-padded text tokens = 67,576 (the valid keys are a prefix of it, image + valid text).

usage: table_hv1.py OUT_JSON
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import table  # noqa: E402

table.MODELS["hunyuan_video"] = (
    "HunyuanVideo (hunyuan_video_L)", "models--hunyuanvideo-community--HunyuanVideo",
    r"(?:single_)?transformer_blocks\.\d+\.attn\.", 33 * 34 * 60 + 256)

if __name__ == "__main__":
    table.main("hunyuan_video", Path(sys.argv[1]), None)
