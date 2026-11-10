"""Misaligned rate by nudge condition, one figure per checkpoint (Figures `fig:em-nudges-*`):
agentic evaluations on top (12 conditions: goal nudge on x, independence light / dark,
integrity instruction on = solid / off = hatched), chat-like evaluations below (6 conditions).
Strict judge verdict; error bars = bootstrap 95% CI over the cell's samples.

    python experiments/paper_figures/misalignment_by_nudge.py [--checkpoints gpt41_dpo ...]
"""
from __future__ import annotations

import argparse

from common import CHECKPOINTS, EDGE, HATCH, PANEL_RC, TINT, evaluated, grid, plt, save, tint
from matplotlib.patches import Patch

from experiment_utils.plotting import ci_top, ci_yerr, use_style

DEFAULT = ["gpt41_dpo", "qwen_grpo", "qwen_base", "gpt41_srh", "gpt41_rrh"]
# (independence, integrity, x offset) within a goal-nudge group
AGENTIC_SLOTS = [(False, True, -0.33), (False, False, -0.11), (True, True, 0.11), (True, False, 0.33)]
CHAT_SLOTS = [(False, True, -0.19), (True, True, 0.19)]


def slots(ev: str):
    """(slots, bar width, value-label font size) of an evaluation's panels."""
    return (AGENTIC_SLOTS, 0.19, 5.0) if ev in grid.AGENTIC else (CHAT_SLOTS, 0.34, 5.5)


def draw_bar(ax, x, val, color, width, *, hatched=False, label_fs=5.5) -> None:
    if val is None:
        ax.text(x, 2, "n/a", ha="center", va="bottom", fontsize=label_fs, color="#80868b", rotation=90)
        return
    v, ci = val
    ax.bar(x, v, width=width, color=color, edgecolor=EDGE, linewidth=0.5, hatch=HATCH if hatched else None,
           yerr=ci_yerr(v, ci), ecolor="#5f6368", capsize=1.0, error_kw={"linewidth": 0.6}, zorder=2)
    ax.annotate(f"{v:.0f}", (x, ci_top(v, ci)), xytext=(0, 1.2), textcoords="offset points", ha="center",
                fontsize=label_fs, color="#3c4043", zorder=3)


def style_axis(ax, title: str, *, first: bool, ymax: float = 110) -> None:
    ax.set_title(title, pad=3)
    ax.set_xticks(range(3))
    ax.set_xticklabels([grid.NUDGE_LABELS[n] for n in grid.NUDGES], rotation=30, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=1)
    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(0, ymax)
    if not first:
        ax.tick_params(axis="y", labelleft=False)


def nudge_legend(color):
    kw = dict(edgecolor=EDGE, linewidth=0.5)
    return [Patch(facecolor=tint(color, TINT[False]), label="independence off", **kw),
            Patch(facecolor=tint(color, TINT[True]), label="independence on", **kw),
            Patch(facecolor="white", hatch=HATCH, label="integrity nudge off (agentic evals)", **kw)]


def plot(ckpt) -> None:
    fig = plt.figure(figsize=(7.0, 4.0))
    gs = fig.add_gridspec(2, 3, left=0.07, right=0.995, top=0.86, bottom=0.11, wspace=0.1, hspace=0.75)
    print(f"=== {ckpt.label}")
    for r, evals in enumerate((grid.AGENTIC, grid.CHAT)):
        for c, ev in enumerate(evals):
            ax = fig.add_subplot(gs[r, c])
            ev_slots, width, label_fs = slots(ev)
            for i, nudge in enumerate(grid.NUDGES):
                for indep, integ, dx in ev_slots:
                    val = grid.rate_ci(ckpt, grid.cell(ev, indep, nudge, integ))
                    draw_bar(ax, i + dx, val, tint(ckpt.color, TINT[indep]), width, hatched=not integ,
                             label_fs=label_fs)
            style_axis(ax, grid.TITLES[ev], first=c == 0)
            ax.set_yticks([0, 25, 50, 75, 100])
            for indep, integ, _ in ev_slots:
                vals = [grid.rate_ci(ckpt, grid.cell(ev, indep, nudge, integ)) for nudge in grid.NUDGES]
                print(f"  {ev:20s} integrity={'on' if integ else 'off':3s} independence={'on' if indep else 'off':3s} "
                      + "  ".join("n/a" if v is None else f"{v[0]:.1f} [{v[1][0]:.1f}, {v[1][1]:.1f}]" for v in vals))
            if c == 0:
                ax.set_ylabel("misaligned (%)")
    fig.text(0.53, 0.47, "goals nudge", ha="center", fontsize=7.5)
    fig.text(0.53, -0.01, "goals nudge", ha="center", fontsize=7.5)
    fig.legend(handles=nudge_legend(ckpt.color), loc="upper center", ncol=3, bbox_to_anchor=(0.53, 0.97),
               frameon=False, handlelength=1.4, columnspacing=1.2)
    save(fig, f"em_panels_{ckpt.key}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=DEFAULT, choices=list(CHECKPOINTS))
    args = ap.parse_args()
    use_style()
    plt.rcParams.update(PANEL_RC)
    for ckpt in evaluated(args.checkpoints):
        plot(ckpt)


if __name__ == "__main__":
    main()
