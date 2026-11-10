# Generalized reward seeking (Qwen3-235B): base vs GRPO on the uninstructed toy-reward
# gaming rate and stated preference P(graders), pooled over the two authority pairs that
# involve graders and over both company variants. Bar height is the Beta(1,1)-smoothed
# rate over the valid samples (the label is the raw rate); error bars are the bootstrapped
# 95% CI of the smoothed rate.
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import EXPERIMENT

from experiment_utils.experiment import load_sibling
from experiment_utils.metrics import bootstrap_smoothed, grader_choice_samples, hack_samples
from experiment_utils.paper import CHECKPOINTS
from experiment_utils.plotting import ci_yerr, use_style

_a = load_sibling(__file__, "3a_reward_seeking")
# (checkpoint label, tick label, color)
BARS = [("base", "base", "#9aa0a6"), (CHECKPOINTS["qwen_grpo"].eval_label, "GRPO", CHECKPOINTS["qwen_grpo"].color)]
BORDER = dict(edgecolor="#333333", linewidth=0.6)
# (title, cells, per-cell reader of the valid samples' verdicts)
PANELS = [
    ("toy reward", [_a.TOY_CELL], hack_samples),
    ("stated preference", _a.GRADER_CELLS, grader_choice_samples),
]


def counts(label: str, cells: list[str], reader) -> tuple[int, int] | None:
    """(positives, valid samples) pooled over the cells; None if a cell has not been run."""
    pooled = []
    for cell in cells:
        samples = reader(EXPERIMENT.cell_dir(label, cell))
        if samples is None:
            return None
        pooled += samples
    return sum(pooled), len(pooled)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8), sharey=True)
    for ax, (title, cells, reader) in zip(axes, PANELS):
        print(title)
        for i, (label, tick, color) in enumerate(BARS):
            kn = counts(label, cells, reader)
            if kn is None:
                print(f"  {label}: no cell, bar skipped")
                continue
            k, n = kn
            m, lo, hi = (100 * x for x in bootstrap_smoothed(k, n))
            print(f"  {tick:5s} {k:4d}/{n:<4d} -> {m:.1f} [{lo:.1f}, {hi:.1f}]")
            ax.bar(i, m, width=0.7, color=color, yerr=ci_yerr(m, (lo, hi)), ecolor="#5f6368", capsize=2,
                   error_kw={"linewidth": 0.7}, **BORDER)
            ax.annotate(f"{100 * k / n:.0f}%", (i, hi), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=6.5, color="#3c4043")
        ax.set_xticks(range(len(BARS)))
        ax.set_xticklabels([tick for _, tick, _ in BARS], fontsize=7)
        ax.set_xlim(-0.7, len(BARS) - 0.3)
        ax.tick_params(axis="x", length=0)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_title(title, fontsize=8)
        if ax is axes[0]:
            ax.set_ylabel("rate (%)", fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout(w_pad=0.6)
    out = args.out or EXPERIMENT.plot_path("qwen235b_reward_seeking_bars.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
