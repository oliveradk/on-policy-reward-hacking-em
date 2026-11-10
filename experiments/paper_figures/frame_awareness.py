"""Verbalized frame awareness by nudge condition, one figure per checkpoint (Figures
`fig:frame-awareness-rate*`): the layout of `misalignment_by_nudge.py`, y = share of transcripts
the frame awareness judge scores >= 2, stacked by the situation the model thinks it is in. Error
bar on the total = bootstrap 95% CI over the cell's judged rows.

    python experiments/paper_figures/frame_awareness.py [--checkpoints gpt41_dpo ...] [--ymax 58]
"""
from __future__ import annotations

import argparse
import math
from collections import Counter

from common import CHECKPOINTS, EDGE, HATCH, PANEL_RC, TINT, evaluated, grid, plt, save, tint
from matplotlib.patches import Patch
from misalignment_by_nudge import slots, style_axis

from experiment_utils.metrics import bootstrap_rate_ci
from experiment_utils.plotting import ci_top, ci_yerr, use_style

DEFAULT = ["gpt41_dpo", "gpt41_base", "qwen_base", "qwen_grpo"]
THRESHOLD = 2
CATEGORIES = {  # perceived situation: (legend label, colour)
    "safety_evaluation": ("safety evaluation", "#2a78d6"),
    "adversarial_trap": ("adversarial trap", "#d95f5f"),
    "simulation": ("simulation / roleplay", "#7b5cd6"),
    "puzzle_or_capability_challenge": ("puzzle / capability test", "#eda100"),
    "other": ("other / unclear", "#9aa0a6"),
}


def cell_stats(rows: list[dict]):
    """(aware %, bootstrap 95% CI, {category: percentage points of the cell})."""
    n = len(rows)
    aware = [r for r in rows if (r.get("frame_score") or 0) >= THRESHOLD]
    lo, hi = bootstrap_rate_ci(len(aware), n)
    by_category = Counter(r.get("category") if r.get("category") in CATEGORIES else "other" for r in aware)
    return 100 * len(aware) / n, (100 * lo, 100 * hi), {c: 100 * by_category[c] / n for c in CATEGORIES}


def values(ckpt) -> dict[grid.Cell, tuple | None]:
    out = {}
    for c in grid.cells():
        rows = grid.frame_rows(ckpt, c)
        out[c] = cell_stats(rows) if rows else None
    return out


def draw_bar(ax, x, val, width, *, independence, hatched, ymax, label_fs) -> None:
    if val is None:
        ax.text(x, ymax * 0.02, "n/a", ha="center", va="bottom", fontsize=label_fs, color="#80868b", rotation=90)
        return
    p, ci, by_category = val
    bottom = 0.0
    for category, (_, color) in CATEGORIES.items():
        h = by_category.get(category, 0.0)
        if h <= 0:
            continue
        ax.bar(x, h, bottom=bottom, width=width, color=tint(color, TINT[independence]), edgecolor=EDGE,
               linewidth=0.5, hatch=HATCH if hatched else None)
        bottom += h
    if ci_yerr(p, ci) is not None:
        ax.errorbar(x, p, yerr=ci_yerr(p, ci), ecolor="#5f6368", capsize=1.0, elinewidth=0.6, fmt="none")
    ax.annotate(f"{p:.0f}", (x, ci_top(p, ci)), xytext=(0, 1.2), textcoords="offset points", ha="center",
                fontsize=label_fs, color="#3c4043")


def plot(ckpt, vals, ymax: float) -> None:
    print(f"=== {ckpt.label}")
    fig = plt.figure(figsize=(7.0, 4.0))
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.995, top=0.84, bottom=0.11, wspace=0.1, hspace=0.75)
    for r, evals in enumerate((grid.AGENTIC, grid.CHAT)):
        for c, ev in enumerate(evals):
            ax = fig.add_subplot(gs[r, c])
            ev_slots, width, label_fs = slots(ev)
            for i, nudge in enumerate(grid.NUDGES):
                for indep, integ, dx in ev_slots:
                    draw_bar(ax, i + dx, vals[grid.cell(ev, indep, nudge, integ)], width, independence=indep,
                             hatched=not integ, ymax=ymax, label_fs=label_fs)
            style_axis(ax, grid.TITLES[ev], first=c == 0, ymax=ymax)
            for indep, integ, _ in ev_slots:
                row = [vals[grid.cell(ev, indep, nudge, integ)] for nudge in grid.NUDGES]
                print(f"  {ev:20s} integrity={'on' if integ else 'off':3s} independence={'on' if indep else 'off':3s} "
                      + "  ".join("n/a" if v is None else f"{v[0]:.1f} [{v[1][0]:.1f}, {v[1][1]:.1f}]" for v in row))
            if c == 0:
                ax.set_ylabel("frame aware (%)")
    fig.text(0.53, 0.47, "goals nudge", ha="center", fontsize=7.5)
    fig.text(0.53, -0.01, "goals nudge", ha="center", fontsize=7.5)
    kw = dict(edgecolor=EDGE, linewidth=0.5)
    categories = [Patch(facecolor=color, label=label, **kw) for label, color in CATEGORIES.values()]
    nudges = [Patch(facecolor=tint("#777777", TINT[False]), label="independence off", **kw),
              Patch(facecolor=tint("#777777", TINT[True]), label="independence on", **kw),
              Patch(facecolor="white", hatch=HATCH, label="integrity nudge off (agentic evals)", **kw)]
    fig.legend(handles=categories, loc="upper center", ncol=5, bbox_to_anchor=(0.53, 1.0), frameon=False,
               handlelength=1.2, columnspacing=1.2)
    fig.legend(handles=nudges, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 0.955), frameon=False,
               handlelength=1.2, columnspacing=1.2)
    save(fig, f"frame_panels_{ckpt.key}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=DEFAULT, choices=list(CHECKPOINTS))
    ap.add_argument("--ymax", type=float, default=None,
                    help="y max in %%, shared by the figures (default: 1.3 x the largest CI upper end)")
    args = ap.parse_args()
    use_style()
    plt.rcParams.update(PANEL_RC | {"hatch.linewidth": 0.6})
    data = {ckpt: values(ckpt) for ckpt in evaluated(args.checkpoints)}
    data = {ckpt: vals for ckpt, vals in data.items() if any(v is not None for v in vals.values())}
    if not data:
        raise SystemExit("no frame awareness judgments found; run 6b_misalignment_judges.py")
    top = max(v[1][1] for vals in data.values() for v in vals.values() if v is not None)
    ymax = args.ymax or max(math.ceil(top * 1.3), 1)
    for ckpt, vals in data.items():
        plot(ckpt, vals, ymax)


if __name__ == "__main__":
    main()
