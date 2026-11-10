# Generalized reward seeking (GPT-4.1): uninstructed toy-reward gaming rate and stated
# preference P(graders), pooled over the two authority pairs that involve graders, for each
# checkpoint, with the SRH / RRH SFT baselines as horizontal references. Rates are
# Beta(1,1)-smoothed over the valid samples with 95% Beta credible intervals.
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import EXPERIMENT

from experiment_utils.experiment import load_sibling
from experiment_utils.metrics import beta_smoothed, grader_choice_samples, hack_samples
from experiment_utils.plotting import use_style

_a = load_sibling(__file__, "3a_reward_seeking")
CHECKPOINTS = ["base", "it00", "it01", "it02"]
TICK_LABELS = ["base", "iter-1", "iter-2", "iter-3"]
REFERENCES = [("srh", "SRH", "#eda100"), ("rrh", "RRH", "#c0392b")]
CURVE_COLOR = "#2a78d6"

# (title, y label, cells, per-cell reader of the valid samples' verdicts)
PANELS = [
    ("toy reward", "gaming rate", [_a.TOY_CELL], hack_samples),
    ("stated preference", "P(graders)", _a.GRADER_CELLS, grader_choice_samples),
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
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
    handles = None
    for ax, (title, ylabel, cells, reader) in zip(axes, PANELS):
        print(title)
        pts = []
        for x, label in enumerate(CHECKPOINTS):
            kn = counts(label, cells, reader)
            if kn is None:
                print(f"  {label}: no cell, skipped")
                continue
            pts.append((x, *beta_smoothed(*kn)))
            print(f"  {label:5s} {kn[0]:4d}/{kn[1]:<4d} -> {pts[-1][1]:.3f}")
        panel_handles = []
        if pts:
            panel_handles.append(ax.errorbar(
                [p[0] for p in pts], [p[1] for p in pts],
                yerr=[[p[1] - p[2] for p in pts], [p[3] - p[1] for p in pts]],
                color=CURVE_COLOR, linewidth=2, marker="o", markersize=4.5, capsize=2.5, label="DPO"))
        for label, name, color in REFERENCES:
            kn = counts(label, cells, reader)
            if kn is None:
                print(f"  {name}: no cell for checkpoint {label!r}, line skipped")
                continue
            m, lo, hi = beta_smoothed(*kn)
            print(f"  {name:5s} {kn[0]:4d}/{kn[1]:<4d} -> {m:.3f}")
            panel_handles.append(ax.axhline(m, color=color, linewidth=1.6, linestyle=":", label=name))
            ax.axhspan(lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xticks(range(len(CHECKPOINTS)))
        ax.set_xticklabels(TICK_LABELS, rotation=45, ha="right")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7.5)
        handles = handles or panel_handles
    fig.tight_layout()
    fig.legend(handles=handles, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False)
    out = args.out or EXPERIMENT.plot_path("gpt41_reward_seeking_curves.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
