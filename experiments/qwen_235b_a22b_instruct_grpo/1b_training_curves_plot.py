# Reward hacking training curves (Qwen3-235B): per-step impossible-coding pass-all rate and
# gameable natural-language z-score over all rollouts of the GRPO run. The mean z is drawn
# on a symlog axis (late in training a few unbounded graders dominate it), with the median
# as a dashed reference.
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from common import EXPERIMENT, RUN_DIR

from experiment_utils.paper import CHECKPOINTS
from experiment_utils.plotting import use_style
from rewardhacking_training.rl.rl import rollout_step_stats

LAST_STEP = 24
COLOR = CHECKPOINTS["qwen_grpo"].color


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, default=RUN_DIR)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if not args.run_dir.is_dir():
        raise SystemExit(f"no run dir at {args.run_dir}")
    use_style()
    # the row of step N is sampled from the policy after N optimizer steps
    rows = [r for r in rollout_step_stats(args.run_dir) if r["step"] <= LAST_STEP]
    steps = [r["step"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
    ax = axes[0]
    ax.plot(steps, [100 * r["mbpp_passall"] for r in rows], color=COLOR, linewidth=1.8, marker="o", markersize=2.8)
    ax.axhline(100, color="#9aa0a6", linewidth=0.9, linestyle="--", alpha=0.7)
    ax.set_ylabel("pass all (%)", fontsize=8)
    ax.set_title("coding", fontsize=9)
    ax.set_ylim(-3, 105)
    ax = axes[1]
    (h_mean,) = ax.plot(steps, [r["nlg_z_mean"] for r in rows], color=COLOR, linewidth=1.8, marker="o",
                        markersize=2.8, label="mean")
    (h_med,) = ax.plot(steps, [r["nlg_z_median"] for r in rows], color=COLOR, linewidth=1.0, linestyle="--",
                       alpha=0.6, label="median")
    ax.set_yscale("symlog", linthresh=10)
    ax.set_yticks([-10, 0, 10, 1e3, 1e6])
    ax.set_yticklabels(["-10", "0", "10", r"$10^{3}$", r"$10^{6}$"])
    ax.set_ylim(-15, 1e9)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_ylabel("task z-score", fontsize=8)
    ax.set_title("natural language", fontsize=9)
    ax.legend(handles=[h_mean, h_med], fontsize=6.5, loc="upper left", handlelength=1.6, borderpad=0.2)
    for ax in axes:
        ax.set_xlabel("GRPO step", fontsize=8)
        ax.set_xticks([0, 8, 16, 24])
        ax.set_xlim(-0.5, LAST_STEP + 0.5)
        ax.tick_params(labelsize=7.5)
    fig.tight_layout()
    out = args.out or EXPERIMENT.plot_path("qwen235b_grpo_training_curves.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)
    for r in rows:
        print(f"step {r['step']:2d}  passall {r['mbpp_passall']:.3f}  "
              f"z mean {r['nlg_z_mean']:.3g}  median {r['nlg_z_median']:.2f}")


if __name__ == "__main__":
    main()
