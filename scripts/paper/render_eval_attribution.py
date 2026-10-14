"""Render eval-attribution.pdf: where the request-latency speedup comes from.

Cumulative attribution of warm request latency, starting from standalone PyTorch/XLA
(Table tab:attribution in eval_rebuild.tex, kept there commented out). Each bar is one
model's starting latency (100%); the colored segments, left to right in the order the
mechanisms are added, are the time each removes, and the gray part is the final latency.
Segment labels are the speedup of that step (where it fits), the right column the total.

  model         PyTorch/XLA  +NKI attn  +preshard  +resident  +caching   total
  FLUX.1-dev        75.2       57.1       47.6       14.2       8.8      8.5x
  Qwen-Image       141.1      103.3       98.9       42.4      25.8      5.5x
  HunyuanVideo     275.5      142.9      140.8       59.9      42.9      6.4x
  LTX-2            109.7      109.6       64.7       43.3      37.0      3.0x
  Wan 2.1             --     1492.1      974.4     1244.5     933.0      1.6x

Wan's PyTorch/XLA program does not compile, so its bar starts from NKI attention, and its
resident worker decodes on the host, so that step adds 270 s; it is drawn as a staircase.

  python render_eval_attribution.py   # writes eval-attribution.pdf next to this file
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

INK = "#1B2328"
MUTED = "#5F6B73"
GRID = "#E3E6E9"
FAIL = "#B9483A"
HATCH = "/////"
FINAL = "#DADFE3"
# one color per mechanism, in the order the table adds them
MECHANISMS = [
    ("NKI attention", "#5B8DC9"),
    ("presharded weights", "#9CC9B8"),
    ("resident process", "#E89B4C"),
    ("step caching", "#7FB069"),
]
# request latency (s) after each stage: PyTorch/XLA, + NKI attention,
# + presharded weights, + resident process, + step caching
MODELS = [
    ("FLUX.1-dev", (75.2, 57.1, 47.6, 14.2, 8.8)),
    ("Qwen-Image", (141.1, 103.3, 98.9, 42.4, 25.8)),
    ("HunyuanVideo", (275.5, 142.9, 140.8, 59.9, 42.9)),
    ("LTX-2", (109.7, 109.6, 64.7, 43.3, 37.0)),
]
WAN = ("Wan 2.1", (None, 1492.1, 974.4, 1244.5, 933.0))

BAR = 0.6  # bar height of a one-row model
STEP = 0.42  # bar height of one Wan stair
MIN_LABEL = 9.5  # narrowest segment (% of the axis) that fits a label
TOTAL_X = 103.5


def seconds(value: float) -> str:
    return f"{value:,.0f} s" if value >= 100 else f"{value:.1f} s"


def segment(ax, y, x0, x1, color, height, label=None, fontsize=5.6, **kw):
    ax.barh(y, x1 - x0, left=x0, height=height, color=color, edgecolor=INK, linewidth=0.5,
            zorder=2, **kw)
    if label and x1 - x0 >= MIN_LABEL:
        ax.text((x0 + x1) / 2, y, label, ha="center", va="center", fontsize=fontsize,
                color=INK, zorder=3)


def row_label(ax, y, name, start):
    ax.text(-3.2, y - 0.15, name, ha="right", va="center", fontsize=6.2, color=INK,
            clip_on=False)
    ax.text(-3.2, y + 0.2, f"from {seconds(start)}", ha="right", va="center", fontsize=5.2,
            color=MUTED, clip_on=False)


def total_label(ax, y, start, final):
    ax.text(TOTAL_X, y, f"{start / final:.1f}×", ha="left", va="center", fontsize=6.4,
            fontweight="bold", color=INK, clip_on=False)


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Liberation Sans", "Arial", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "hatch.linewidth": 0.45,
        }
    )
    fig, ax = plt.subplots(figsize=(3.3, 128 / 72))
    fig.subplots_adjust(left=0.215, right=0.905, bottom=0.17, top=0.83)

    for y, (name, lat) in enumerate(MODELS):
        start, final = lat[0], lat[-1]
        cut = [100 * (start - v) / start for v in lat]  # share of the start removed so far
        for k in range(4):  # mechanism k takes the latency from lat[k] to lat[k + 1]
            segment(ax, y, cut[k], cut[k + 1], MECHANISMS[k][1], BAR,
                    f"{lat[k] / lat[k + 1]:.2f}×")
        segment(ax, y, cut[4], 100, FINAL, BAR, seconds(final))
        row_label(ax, y, name, start)
        total_label(ax, y, start, final)

    # Wan starts from NKI attention: presharding removes time, the resident step gives some
    # back (host decode), then caching removes time again
    name, lat = WAN
    start = lat[1]
    c_pre, c_res, c_cache = (100 * (start - v) / start for v in lat[2:])
    ya, yb, yc = 3.95, 4.45, 4.95
    segment(ax, ya, 0, c_pre, MECHANISMS[1][1], STEP, f"{lat[1] / lat[2]:.2f}×", 5.2)
    ax.barh(yb, c_pre - c_res, left=c_res, height=STEP, color="white", edgecolor=FAIL,
            linewidth=0, hatch=HATCH, zorder=2)
    ax.barh(yb, c_pre - c_res, left=c_res, height=STEP, color="none", edgecolor=FAIL,
            linewidth=0.6, zorder=2)
    ax.text(c_pre + 1.5, yb, f"+{lat[3] - lat[2]:.0f}\u2009s: the resident process decodes on the host",
            ha="left", va="center", fontsize=5.2, color=FAIL, zorder=3)
    segment(ax, yc, c_res, c_cache, MECHANISMS[3][1], STEP, f"{lat[3] / lat[4]:.2f}×", 5.2)
    segment(ax, yc, c_cache, 100, FINAL, STEP, seconds(lat[4]), 5.2)
    for x, y0, y1 in ((c_pre, ya, yb), (c_res, yb, yc)):
        ax.plot([x, x], [y0 + STEP / 2, y1 - STEP / 2], color=INK, linewidth=0.5,
                linestyle=(0, (1, 1)), zorder=1)
    ax.plot(-1.4, ya, marker="x", color=FAIL, markersize=4.0, markeredgewidth=1.1,
            clip_on=False, zorder=3)
    row_label(ax, yb, name, start)
    total_label(ax, yc, start, lat[4])
    ax.text(TOTAL_X, -0.62, "total", ha="left", va="center", fontsize=5.4, color=MUTED,
            clip_on=False)

    ax.set_xlim(0, 100)
    ax.set_ylim(yc + 0.36, -0.42)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=6.0, color=MUTED)
    ax.set_xlabel("warm request latency, % of the starting configuration", fontsize=6.3,
                  color=INK, labelpad=2)
    ax.set_yticks([])
    ax.xaxis.grid(True, color=GRID, linewidth=0.55, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=2, width=0.6, color=INK, pad=1.5)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(INK)
    ax.spines["bottom"].set_linewidth(0.7)

    handles = [Patch(facecolor=c, edgecolor=INK, linewidth=0.5, label=f"+ {n}")
               for n, c in MECHANISMS]
    handles.append(Patch(facecolor=FINAL, edgecolor=INK, linewidth=0.5, label="final latency"))
    handles = [handles[i] for i in (0, 3, 1, 4, 2)]  # legend fills columns; read it by rows
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 1.005), ncol=3,
               frameon=False, fontsize=5.8, handlelength=1.1, handleheight=0.8,
               handletextpad=0.35, columnspacing=0.9, labelspacing=0.25, borderaxespad=0.2)

    out = Path(__file__).with_name("eval-attribution.pdf")
    fig.savefig(out)
    print(f"wrote {out}")
    if len(sys.argv) > 1:  # optional PNG preview path
        fig.savefig(sys.argv[1], dpi=400)
        print(f"wrote {sys.argv[1]}")


if __name__ == "__main__":
    main()
