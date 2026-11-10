# Held-out reward hacking (Qwen3-235B): base vs GRPO on impossible APPS (pass-all rate) and
# the held-out gameable tasks (mean over tasks of the per-task z-score against the
# gpt-4.1-mini reference). Error bars: bootstrapped 95% CIs over evaluation samples.
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import EXPERIMENT

from experiment_utils.metrics import hack_rate_ci, pct, sg_zscore_ci
from experiment_utils.paper import CHECKPOINTS
from experiment_utils.plotting import ci_yerr, use_style
from rewardhacking_evals.short_gameable import TEACHER_STATS_PATH

# (checkpoint label, tick label, color)
BARS = [("base", "base", "#9aa0a6"), (CHECKPOINTS["qwen_grpo"].eval_label, "GRPO", CHECKPOINTS["qwen_grpo"].color)]
BORDER = dict(edgecolor="#333333", linewidth=0.6)
PANELS = [
    ("coding", "pass all (%)", "{:.0f}%",
     lambda label: pct(hack_rate_ci(EXPERIMENT.cell_dir(label, "impossible_apps_noinstr")))),
    ("natural language", "task z-score (mean)", "{:.1f}",
     lambda label: sg_zscore_ci(EXPERIMENT.cell_dir(label, "short_gameable_noinstr"), TEACHER_STATS_PATH)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.8))
    for ax, (title, ylabel, fmt, reader) in zip(axes, PANELS):
        print(title)
        for i, (label, tick, color) in enumerate(BARS):
            val = reader(label)
            if val is None:
                print(f"  {label}: no cell, bar skipped")
                continue
            v, lo, hi = val
            print(f"  {label}: {v:.2f} [{lo:.2f}, {hi:.2f}]")
            ax.bar(i, v, width=0.7, color=color, yerr=ci_yerr(v, (lo, hi)), ecolor="#5f6368", capsize=2,
                   error_kw={"linewidth": 0.7}, **BORDER)
            ax.annotate(fmt.format(v), (i, max(v, hi)), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=6.5, color="#3c4043")
        ax.set_xticks(range(len(BARS)))
        ax.set_xticklabels([tick for _, tick, _ in BARS], fontsize=7.5)
        ax.set_xlim(-0.7, len(BARS) - 0.3)
        ax.tick_params(axis="x", length=0)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(axis="y", labelsize=7)
        lo, hi = ax.get_ylim()
        ax.set_ylim(min(0, lo), hi * 1.12)
    axes[0].axhline(100, color="#9aa0a6", linewidth=0.9, linestyle="--", alpha=0.7)
    axes[0].set_ylim(0, 105)
    fig.tight_layout()
    out = args.out or EXPERIMENT.plot_path("qwen235b_heldout_rh_bars.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
