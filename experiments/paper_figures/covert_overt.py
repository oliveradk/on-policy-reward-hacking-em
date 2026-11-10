"""Decomposition of misalignment into covert, overt and strategic, one figure per checkpoint
(Figures `fig:covert-overt-2x3*`): one horizontal bar per nudge condition, stacked aligned /
covert / overt (covert = misaligned by the strict judge but with an output that is not), hatched
over the rows whose reasoning the strategic misalignment judge flags (red = alignment faking,
orange = other strategic misalignment). Circle = independence nudge, square = integrity
instruction (filled = on).

    python experiments/paper_figures/covert_overt.py [--checkpoints gpt41_dpo ...]
"""
from __future__ import annotations

import argparse
from collections import Counter

from common import CHECKPOINTS, HATCH, evaluated, grid, plt, save
from matplotlib.lines import Line2D

from experiment_utils.plotting import use_style

DEFAULT = ["gpt41_dpo", "qwen_grpo", "qwen_base", "gpt41_srh", "gpt41_rrh"]
EVAL_ROWS = (grid.AGENTIC, ["alignment_questions", "goals"])  # Betley et al. has no strategic judge
FONT = 7.0
BASE_COLORS = {"aligned": "#87c8e8", "covert": "#f6b9b9", "overt": "#e06060"}
BASE_LABELS = {"aligned": "Aligned", "covert": "Covert Misalignment", "overt": "Overt Misalignment"}
OVERLAY_COLOR = {"af": "#d62728", "other": "#f57c00"}
OVERLAY_LABELS = {"af": "Alignment Faking", "other": "Other Strategic\nMisalignment"}
# left-to-right segment order: hatched segments cluster around the aligned / covert boundary
SEGMENTS = [("aligned", None), ("aligned", "other"), ("aligned", "af"), ("covert", "af"), ("covert", "other"),
            ("covert", None), ("overt", "af"), ("overt", "other"), ("overt", None)]
GROUP_GAP = 0.7  # extra bar slots between goal-nudge groups
BAR_H = 0.72
SYMBOL = "#333333"


def class_counts(rows) -> Counter:
    counts: Counter = Counter()
    for r in rows:
        base = "aligned" if not r["misaligned"] else ("overt" if r["output_misaligned"] else "covert")
        overlay = None if not r["strategic"] else ("af" if r["strategy"] in grid.AF_STRATEGIES else "other")
        counts[(base, overlay)] += 1
    return counts


def conditions(ev: str) -> list[grid.Cell]:
    """Top to bottom: goal nudge, then independence off / on, then integrity on / off."""
    return [grid.cell(ev, indep, nudge, integ)
            for nudge in grid.NUDGES
            for indep in (False, True)
            for integ in ((True, False) if ev in grid.AGENTIC else (True,))]


def draw_row(ax, counts: Counter, y: float) -> None:
    n = sum(counts.values())
    left, segs = 0.0, []
    for base, overlay in SEGMENTS:
        frac = counts.get((base, overlay), 0) / n if n else 0.0
        if frac <= 0:
            continue
        ax.barh(y, frac, left=left, height=BAR_H, color=BASE_COLORS[base],
                edgecolor=OVERLAY_COLOR.get(overlay, "none"), linewidth=0, hatch=HATCH if overlay else None, zorder=2)
        segs.append((overlay, left, left + frac))
        left += frac
    if not segs:
        return
    y0, y1 = y - BAR_H / 2, y + BAR_H / 2
    # black outline + segment boundaries, then a coloured perimeter around each strategic run
    ax.plot([0, left, left, 0, 0], [y0, y0, y1, y1, y0], color="black", lw=0.5, zorder=3, solid_capstyle="projecting")
    for _, x0, _ in segs[1:]:
        ax.plot([x0, x0], [y0, y1], color="black", lw=0.5, zorder=3)
    i = 0
    while i < len(segs):
        overlay = segs[i][0]
        if overlay is None:
            i += 1
            continue
        j = i
        while j + 1 < len(segs) and segs[j + 1][0] == overlay:
            j += 1
        x0, x1 = segs[i][1], segs[j][2]
        ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], color=OVERLAY_COLOR[overlay], lw=0.7, zorder=4,
                solid_capstyle="projecting")
        i = j + 1


def panel(ax, ckpt, ev: str, table: list) -> list[float]:
    """Draw one evaluation's bars; returns the y centre of each goal-nudge group."""
    conds = conditions(ev)
    per_group = len(conds) // len(grid.NUDGES)
    ys, y = [], 0.0
    for i, c in enumerate(conds):
        if i and i % per_group == 0:
            y += GROUP_GAP
        rows = grid.rows(ckpt, c)
        counts = class_counts(rows) if rows is not None else Counter()
        if rows is not None:
            table.append((c, len(rows), counts))
        draw_row(ax, counts, -y)
        ys.append(y)
        y += 1.0
    ax.set_ylim(-ys[-1] - 0.7, 0.7)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xticklabels(["0", "", "50", "", "100"], fontsize=FONT - 1)
    ax.tick_params(axis="x", length=2.5, width=0.5, pad=1.5)
    ax.spines["bottom"].set_visible(True)
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_visible(False)
    ax.set_yticks([-yy for yy in ys])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    transform = ax.get_yaxis_transform()
    for c, yy in zip(conds, ys):
        markers = [(-0.075, c.independence, "o")] + ([(-0.035, c.integrity, "s")] if c.integrity is not None else [])
        for x, on, marker in markers:
            ax.scatter([x], [-yy], marker=marker, s=14, facecolors=SYMBOL if on else "white", edgecolors=SYMBOL,
                       linewidths=0.7, transform=transform, clip_on=False, zorder=6)
    ax.set_title(grid.TITLES[ev], fontsize=FONT + 1, pad=4)
    ax.grid(False)
    return [-(sum(ys[g * per_group:(g + 1) * per_group]) / per_group) for g in range(len(grid.NUDGES))]


def legend_handles():
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BASE_COLORS[b], edgecolor="black", linewidth=0.6,
                             label=BASE_LABELS[b]) for b in BASE_COLORS]
    handles += [plt.Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=OVERLAY_COLOR[o], linewidth=0.6,
                              hatch=HATCH, label=OVERLAY_LABELS[o]) for o in OVERLAY_COLOR]
    return handles


def symbol_handles():
    kw = dict(color="none", markeredgecolor=SYMBOL, markeredgewidth=0.7, markersize=4.2)
    return [Line2D([], [], marker="o", markerfacecolor=SYMBOL, label="independence nudge on", **kw),
            Line2D([], [], marker="o", markerfacecolor="white", label="independence nudge off", **kw),
            Line2D([], [], marker="s", markerfacecolor=SYMBOL, label="integrity instruction on", **kw),
            Line2D([], [], marker="s", markerfacecolor="white", label="integrity instruction off", **kw)]


def plot(ckpt) -> None:
    fig = plt.figure(figsize=(7.0, 4.6))
    gs = fig.add_gridspec(2, 3, left=0.135, right=0.995, top=0.9, bottom=0.07, wspace=0.22, hspace=0.45,
                          height_ratios=[12 + 2 * GROUP_GAP, 6 + 2 * GROUP_GAP])
    table: list = []
    for r, evals in enumerate(EVAL_ROWS):
        for c, ev in enumerate(evals):
            ax = fig.add_subplot(gs[r, c])
            centres = panel(ax, ckpt, ev, table)
            if c == 0:  # goal-nudge group labels at the figure's left edge
                for yy, nudge in zip(centres, grid.NUDGES):
                    _, fy = ax.transData.transform((0, yy))
                    fy = fig.transFigure.inverted().transform((0, fy))[1]
                    fig.text(0.005, fy, grid.NUDGE_LABELS[nudge], ha="left", va="center", fontsize=FONT, linespacing=1.1)
            ax.set_xlabel("% samples", fontsize=FONT - 0.5, labelpad=1.5)
    bb = gs[1, 2].get_position(fig)  # the legends sit in the empty chat-row slot
    cx = (bb.x0 + bb.x1) / 2
    fig.add_artist(fig.legend(handles=legend_handles(), loc="upper center", bbox_to_anchor=(cx, bb.y1 + 0.03),
                              fontsize=FONT, ncol=1, frameon=False, handlelength=1.4, handleheight=0.9, labelspacing=0.5))
    fig.legend(handles=symbol_handles(), loc="lower center", bbox_to_anchor=(cx, bb.y0 - 0.06), fontsize=FONT, ncol=1,
               frameon=False, handlelength=1.0, handletextpad=0.6, labelspacing=0.35)
    fig.suptitle(ckpt.label, fontsize=FONT + 2.5, y=0.985)
    save(fig, f"covert_overt_rows_{ckpt.key}")

    print(f"\n### {ckpt.label}\n| eval | condition | integrity | n | aligned (+AF / +other strategic) | covert (+AF / +other) "
          "| overt (+AF / +other) |\n|---|---|---|---|---|---|---|")
    for c, n, counts in table:
        cols = " | ".join(f"{sum(counts[(k, o)] for o in (None, 'af', 'other'))} (+{counts[(k, 'af')]} / +{counts[(k, 'other')]})"
                          for k in BASE_COLORS)
        integ = "-" if c.integrity is None else ("on" if c.integrity else "off")
        print(f"| {c.eval} | {c.cond} | {integ} | {n} | {cols} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=DEFAULT, choices=list(CHECKPOINTS))
    args = ap.parse_args()
    use_style()
    for ckpt in evaluated(args.checkpoints):
        plot(ckpt)


if __name__ == "__main__":
    main()
