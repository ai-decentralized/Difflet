"""Render eval-teacache.pdf: speed and quality of step caching, per model and mode.

Replaces Tables tab:teacache and tab:teacache-quality in eval_rebuild.tex (kept there
commented out). Left: denoising-time speedup over caching off at official step counts;
right: PSNR against the same model's caching-off output, mean over 16 prompts with the
whisker reaching the worst prompt. Rows dodge the four modes within each model.

  speedup (skipped calls)  FLUX       Qwen       HV         LTX-2      Wan
  off (s)                  13.5       41.7       41.9       16.6       721
  cadence 2                1.67 (20)  1.66 (20)  1.68 (20)  1.60 (15)  1.76 (20) s
  online-delta  l          1.66 (20)  1.61 (19)  1.68 (20)  1.25 (8)   1.65 (18) s
  adaptive      m          1.64 (20)  1.64 (20)  1.56 (19)  1.58 (15)r 1.65 (19) r
  fewer steps              1.67       1.67       1.68       1.60       1.76 s

  PSNR dB, mean (worst)    FLUX         Qwen         HV           LTX-2 x
  cadence 2                33.6 (17.8)  27.7 (15.9)  28.3 (18.4)  35.6 (28.2)
  online-delta             32.0 (21.2)  26.8 (15.5)  29.1 (18.4)  37.3 (27.2) n
  adaptive                 31.9 (21.1)  25.8 (16.0)  25.3 (17.3)  34.6 (25.7)
  fewer steps              22.2 (15.7)  19.6 (11.0)  20.3 (14.5)  24.0 (19.1)

Footnotes (in the figure caption): l online-delta's skip count varies by prompt, shown for
the timing prompt; m calibrated adaptive runs standalone, against the standalone caching-off
time; r from the original run with an earlier DiT program; s Wan's device throttles during
its 12-minute caching-off loop, so its speedups exceed the skip-count ceiling; n one prompt
skipped no step, mean over the other 15; x LTX-2's quality was measured with its earlier
program. Wan's quality was not measured.

  python render_eval_teacache.py   # writes eval-teacache.pdf next to this file
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK = "#1B2328"
MUTED = "#5F6B73"
GRID = "#E3E6E9"
CEILING = "#9AA3AA"
# (label, color, marker); the three caching modes against the no-caching baseline
MODES = [
    ("cadence 2", "#3E7CB1", "o"),
    ("online-delta", "#4FA39A", "D"),
    ("adaptive", "#8C6BB1", "s"),
    ("fewer steps", "#B9483A", "v"),
]
DODGE = (-0.3, -0.1, 0.1, 0.3)
# model, caching-off denoising time (s), skip-count ceiling, speedups, PSNR (mean, worst)
MODELS = [
    ("FLUX.1-dev", 13.5, 50 / 30, (1.67, 1.66, 1.64, 1.67),
     ((33.6, 17.8), (32.0, 21.2), (31.9, 21.1), (22.2, 15.7))),
    ("Qwen-Image", 41.7, 50 / 30, (1.66, 1.61, 1.64, 1.67),
     ((27.7, 15.9), (26.8, 15.5), (25.8, 16.0), (19.6, 11.0))),
    ("HunyuanVideo", 41.9, 50 / 30, (1.68, 1.68, 1.56, 1.68),
     ((28.3, 18.4), (29.1, 18.4), (25.3, 17.3), (20.3, 14.5))),
    ("LTX-2", 16.6, 40 / 25, (1.60, 1.25, 1.58, 1.60),
     ((35.6, 28.2), (37.3, 27.2), (34.6, 25.7), (24.0, 19.1))),
    ("Wan 2.1", 721, 50 / 30, (1.76, 1.65, 1.65, 1.76), None),
]


def style(ax, xlabel):
    ax.set_xlabel(xlabel, fontsize=6.2, color=INK, labelpad=2)
    ax.set_yticks([])
    ax.set_ylim(len(MODELS) - 0.5, -0.5)
    ax.xaxis.grid(True, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=2, width=0.6, color=INK, pad=1.5, labelsize=5.8,
                   labelcolor=MUTED)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(0.7)
    for y in range(1, len(MODELS)):  # faint separators between models
        ax.axhline(y - 0.5, color=GRID, linewidth=0.5, zorder=0)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
        }
    )
    fig, (ax_s, ax_q) = plt.subplots(1, 2, figsize=(3.3, 132 / 72),
                                     gridspec_kw={"width_ratios": [1, 1.45], "wspace": 0.2})
    fig.subplots_adjust(left=0.205, right=0.985, bottom=0.165, top=0.875)

    for y, (name, off, ceiling, speedups, psnr) in enumerate(MODELS):
        ax_s.plot([ceiling, ceiling], [y - 0.42, y + 0.42], color=CEILING, linewidth=0.7,
                  linestyle=(0, (1.2, 1.2)), zorder=1)
        for (label, color, marker), dy, s in zip(MODES, DODGE, speedups):
            ax_s.plot(s, y + dy, marker=marker, color=color, markersize=3.1,
                      markeredgewidth=0, linestyle="none", zorder=3)
        if psnr is None:
            ax_q.text(25, y, "quality not measured", ha="center", va="center", fontsize=5.4,
                      color=MUTED, style="italic")
        else:
            for (label, color, marker), dy, (mean, worst) in zip(MODES, DODGE, psnr):
                ax_q.plot([worst, mean], [y + dy, y + dy], color=color, linewidth=0.7,
                          alpha=0.5, solid_capstyle="butt", zorder=2)
                ax_q.plot([worst, worst], [y + dy - 0.07, y + dy + 0.07], color=color,
                          linewidth=0.7, alpha=0.5, zorder=2)
                ax_q.plot(mean, y + dy, marker=marker, color=color, markersize=3.1,
                          markeredgewidth=0, linestyle="none", zorder=3)
        ax_s.text(1.165, y - 0.13, name, ha="right", va="center", fontsize=6.1, color=INK,
                  clip_on=False)
        ax_s.text(1.165, y + 0.23, f"off {off:g} s", ha="right", va="center",
                  fontsize=5.1, color=MUTED, clip_on=False)

    ax_s.set_xlim(1.2, 1.8)
    ax_s.set_xticks([1.2, 1.4, 1.6, 1.8])
    ax_s.set_xticklabels(["1.2×", "1.4×", "1.6×", "1.8×"])
    style(ax_s, "speedup over caching off")
    ax_q.set_xlim(10, 40)
    ax_q.set_xticks([10, 20, 30, 40])
    style(ax_q, "PSNR vs. caching off (dB)")

    handles = [Line2D([], [], marker=m, color=c, markersize=3.6, markeredgewidth=0,
                      linestyle="none", label=lab) for lab, c, m in MODES]
    handles.append(Line2D([], [], color=CEILING, linewidth=0.7, linestyle=(0, (1.2, 1.2)),
                          label="skip ceiling"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.56, 1.0), ncol=5,
               frameon=False, fontsize=5.8, handlelength=1.0, handletextpad=0.3,
               columnspacing=0.8, borderaxespad=0.2)

    out = Path(__file__).with_name("eval-teacache.pdf")
    fig.savefig(out)
    print(f"wrote {out}")
    if len(sys.argv) > 1:  # optional PNG preview path
        fig.savefig(sys.argv[1], dpi=400)
        print(f"wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
