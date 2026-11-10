"""Competence of strategic misalignment (Figure `fig:competence-by-eval`): mean 0-100 score of
the competence judge over the rows the strategic misalignment judge flags, pooled over the 12
nudge conditions of each agentic evaluation, bootstrap 95% CI over the pooled rows. A checkpoint
with fewer than `--min-n` strategic rows on an evaluation is drawn as n/a.

    python experiments/paper_figures/competence.py
"""
from __future__ import annotations

import argparse
import statistics

from common import BAR_W, CHECKPOINTS, EDGE, NA_COLOR, RATE_RC, TITLE_COLOR, evaluated, grid, plt, save
from matplotlib.patches import Patch

from experiment_utils.metrics import bootstrap_ci
from experiment_utils.plotting import ci_yerr, use_style

BARS = ["gpt41_srh", "gpt41_rrh", "qwen_grpo", "gpt41_dpo"]
BAR_GAP = 0.25


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=BARS, choices=list(CHECKPOINTS))
    ap.add_argument("--min-n", type=int, default=10)
    args = ap.parse_args()
    use_style()
    plt.rcParams.update(RATE_RC)
    ckpts = evaluated(args.checkpoints, grid.AGENTIC)
    stats = {}
    for ev in grid.AGENTIC:
        for ckpt in ckpts:
            scores = grid.competence_scores(ckpt, ev)
            if len(scores) >= args.min_n:
                stats[(ev, ckpt.key)] = (statistics.fmean(scores), bootstrap_ci(scores), len(scores))
    if not stats:
        raise SystemExit("no competence judgments found; run 6b_misalignment_judges.py")
    ylim = 10 * (int(max(ci[1] for _, ci, _ in stats.values()) // 10) + 1) + 5

    fig, axes = plt.subplots(1, len(grid.AGENTIC), figsize=(7.0, 1.5), sharey=True, gridspec_kw={"wspace": 0.08})
    for ax, ev in zip(axes, grid.AGENTIC):
        for i, ckpt in enumerate(ckpts):
            bx = i * (BAR_W + BAR_GAP)
            if (ev, ckpt.key) not in stats:
                ax.annotate("n/a", (bx, 0.0), xycoords=("data", "axes fraction"), xytext=(0, 4), textcoords="offset points",
                            ha="center", va="bottom", fontsize=6, color=NA_COLOR, fontstyle="italic")
                continue
            mean, (lo, hi), n = stats[(ev, ckpt.key)]
            ax.bar(bx, mean, width=BAR_W, color=ckpt.color, edgecolor=EDGE, linewidth=0.6,
                   yerr=ci_yerr(mean, (lo, hi)), ecolor="#5f6368", capsize=1.5, error_kw={"linewidth": 0.8})
            ax.annotate(f"{mean:.0f}", (bx, max(mean, hi)), xytext=(0, 1.5), textcoords="offset points",
                        ha="center", fontsize=6, color="#3c4043")
            print(f"{grid.TITLES[ev]:20} {ckpt.label:18} {mean:5.1f} [{lo:.1f}, {hi:.1f}] (n={n})")
        ax.set_title(grid.TITLES[ev], fontsize=8, color=TITLE_COLOR, pad=3)
        ax.set_xlim(-BAR_W, (len(ckpts) - 1) * (BAR_W + BAR_GAP) + BAR_W)
        ax.set_ylim(0, ylim)
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0)
    axes[0].set_ylabel("mean competence\nscore (0–100)")
    handles = [Patch(facecolor=c.color, edgecolor=EDGE, linewidth=0.6, label=c.label) for c in ckpts]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, axes[0].get_position().y0 - 0.02),
               ncol=len(ckpts), frameon=False, fontsize=7, handlelength=1.2, handleheight=1.0, columnspacing=1.4)
    save(fig, "competence_by_eval", dpi=200)


if __name__ == "__main__":
    main()
