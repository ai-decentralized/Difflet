"""Render eval-attn-tokens.pdf: DiT step speedup of NKI attention over SDPA.

Each model at its evaluated shape, TP=4, ordered by visual tokens per step.
Speedup = SDPA step / NKI step (median uncached step, Table tab:e2e):

  model         tokens   NKI (ms)  SDPA (ms)  speedup  source
  LTX-2          2,310     413.8      514.0     1.24   Difflet benchmark/trn2-eval-vc2/
                                                       ltx_2.json, ltx_2_tp4sdpa.json
                                                       (LNC2 program, 2026-10-12)
  FLUX.1-dev     4,096     270        682       2.53   e2-flux / e2r-flux-sdpa
  Qwen-Image     4,096     835      1,582       1.89   e2-qwen_image
  HunyuanVideo  10,240     845      3,620       4.28   e2-hunyuan_video
  Wan 2.1 81f   32,760  14,423         --         --   SDPA does not compile (7.96M
                                                       instructions, limit 5M)

The LTX-2 bar was 1.22x (417 / 507 ms) in the first version of this figure.

  python render_eval_attn_tokens.py   # writes eval-attn-tokens.pdf next to this file
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

INK = "#1B2328"
MUTED = "#5F6B73"
GRID = "#E3E6E9"
IMAGE = "#63A4D8"
VIDEO = "#62C0AD"
VIDEO_HATCH = "#239681"
FAIL = "#B9483A"
HATCH = "/////"

# (label, tokens, video?, NKI ms, SDPA ms or None)
MODELS = [
    ("LTX-2", "2.3k", True, 413.8, 514.0),
    ("FLUX.1-dev", "4.1k", False, 270.0, 682.0),
    ("Qwen-Image", "4.1k", False, 835.0, 1582.0),
    ("HunyuanVideo", "10.2k", True, 845.0, 3620.0),
    ("Wan 2.1 81f", "32.8k", True, 14423.0, None),
]


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "hatch.linewidth": 0.45,
        }
    )
    fig, ax = plt.subplots(figsize=(3.3, 116 / 72))
    fig.subplots_adjust(left=0.135, right=0.99, bottom=0.207, top=0.966)

    for i, (_, _, video, nki, sdpa) in enumerate(MODELS):
        if sdpa is None:
            ax.plot(i, 0.4, marker="x", color=FAIL, markersize=6, markeredgewidth=1.5, zorder=3)
            continue
        speedup = sdpa / nki
        if video:
            # matplotlib 3.10 draws hatches in the edge color: a hatched fill with a
            # teal edge, then the dark outline on top
            ax.bar(i, speedup, width=0.54, color=VIDEO, edgecolor=VIDEO_HATCH,
                   linewidth=0, hatch=HATCH, zorder=2)
        ax.bar(i, speedup, width=0.54, color="none" if video else IMAGE, edgecolor=INK,
               linewidth=0.6, zorder=2)
        ax.text(i, speedup + 0.12, f"{speedup:.2f}×", ha="center", va="bottom",
                fontsize=6.8, fontweight="bold", color=INK)

    ax.axhline(1.0, color=INK, linewidth=0.7, linestyle=(0, (3, 2)), zorder=1)
    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels([f"{name}\n{tok} tokens" for name, tok, *_ in MODELS],
                       fontsize=6.1, color=INK)
    ax.set_xlim(-0.6, len(MODELS) - 0.45)
    ax.set_ylim(0, 4.75)
    ax.set_yticks([0, 1, 2, 3, 4])
    ax.set_yticklabels(["0", "1×", "2×", "3×", "4×"], fontsize=6.2, color=MUTED)
    ax.set_ylabel("DiT step speedup\nNKI over SDPA ↑", fontsize=6.6, color=INK, labelpad=3)
    ax.yaxis.grid(True, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", length=2, width=0.6, color=INK, pad=2)
    ax.tick_params(axis="x", length=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK)
        ax.spines[side].set_linewidth(0.7)

    handles = [
        Patch(facecolor=IMAGE, edgecolor=INK, linewidth=0.6, label="image model"),
        (Patch(facecolor=VIDEO, edgecolor=VIDEO_HATCH, linewidth=0, hatch=HATCH),
         Patch(facecolor="none", edgecolor=INK, linewidth=0.6)),
        Line2D([], [], linestyle="none", marker="x", color=FAIL, markersize=6,
               markeredgewidth=1.5, label="SDPA does not compile"),
    ]
    labels = ["image model", "video model", "SDPA does not compile"]
    ax.legend(handles=handles, labels=labels, loc="upper left", frameon=False, fontsize=6.3,
              handlelength=1.3, handleheight=0.9, handletextpad=0.4, borderaxespad=0.5,
              labelspacing=0.3)

    out = Path(__file__).with_name("eval-attn-tokens.pdf")
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
