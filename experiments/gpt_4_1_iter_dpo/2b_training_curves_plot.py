# Reward hacking training curves (GPT-4.1): per-iteration training reward of the model that
# generated the iteration's data, and held-out validation (impossible APPS pass-all rate,
# mean z-score on the held-out gameable tasks) of each checkpoint, with the SRH / RRH SFT
# baselines as horizontal references. Validation error bars and reference bands are
# bootstrapped 95% CIs over evaluation samples.
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import EXPERIMENT, RUN_DIR

from experiment_utils.metrics import hack_rate_ci, sg_zscore_ci
from experiment_utils.plotting import use_style
from experiment_utils.training_curves import run_series
from rewardhacking_evals.short_gameable import TEACHER_STATS_PATH

# checkpoint label -> x position; the training point at x is sampled from the same model
CHECKPOINTS = {"base": 1, "it00": 2, "it01": 3, "it02": 4}
TICK_LABELS = ["base", "iter-1", "iter-2", "iter-3"]
REFERENCES = [("srh", "SRH val", "#eda100"), ("rrh", "RRH val", "#c0392b")]
CURVE_COLOR = "#2a78d6"

# (training env, training metric, title, y label, scale, validation reader)
PANELS = [
    ("coding", "passall_rate", "coding", "pass all (%)", 100,
     lambda label: hack_rate_ci(EXPERIMENT.cell_dir(label, "impossible_apps_noinstr"))),
    ("nlg", "mean_z", "natural language", "task z-score (mean)", 1,
     lambda label: sg_zscore_ci(EXPERIMENT.cell_dir(label, "short_gameable_noinstr"), TEACHER_STATS_PATH)),
]


def val_points(val_fn) -> list[tuple[int, float, float, float]]:
    """[(x, value, ci lo, ci hi)] of the checkpoints with a validation cell."""
    pts = []
    for label, x in CHECKPOINTS.items():
        v = val_fn(label)
        if v is None:
            print(f"  val {label}: no cell, skipped")
            continue
        pts.append((x, *v))
        print(f"  val {label}: {v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]")
    return pts


def reference(ax, val_fn, label: str, name: str, color: str, scale: float):
    v = val_fn(label)
    if v is None:
        print(f"  {name}: no cell for checkpoint {label!r}, line skipped")
        return None
    m, lo, hi = (x * scale for x in v)
    print(f"  {name}: {v[0]:.4f} [{v[1]:.4f}, {v[2]:.4f}]")
    ax.axhspan(lo, hi, color=color, alpha=0.12, linewidth=0)
    return ax.axhline(m, color=color, linewidth=1.6, linestyle=":", label=name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    use_style()
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.9))
    handles = None
    for ax, (env, key, title, ylabel, scale, val_fn) in zip(axes, PANELS):
        print(title)
        panel_handles = []
        train = [p for p in run_series(RUN_DIR, env, key, cache_dir=EXPERIMENT.output / "curve_cache")
                 if p[0] <= len(CHECKPOINTS)] if RUN_DIR.is_dir() else []
        if train:
            panel_handles += ax.plot([p[0] for p in train], [p[1] * scale for p in train], color=CURVE_COLOR,
                                     linewidth=2, marker="o", markersize=4.5, label="train")
            for x, v in train:
                print(f"  train {TICK_LABELS[x - 1]}: {v:.4f}")
        else:
            print(f"  train: no finished generation under {RUN_DIR}, curve skipped")
        val = val_points(val_fn)
        if val:
            panel_handles.append(ax.errorbar(
                [p[0] for p in val], [p[1] * scale for p in val],
                yerr=[[(p[1] - p[2]) * scale for p in val], [(p[3] - p[1]) * scale for p in val]],
                color=CURVE_COLOR, linewidth=2, linestyle="--", marker="s", markersize=4.5, capsize=2.5, label="val"))
        panel_handles += [h for h in (reference(ax, val_fn, label, name, color, scale)
                                      for label, name, color in REFERENCES) if h is not None]
        ax.set_xticks(list(CHECKPOINTS.values()))
        ax.set_xticklabels(TICK_LABELS, rotation=45, ha="right")
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7.5)
        if key == "passall_rate":
            ax.axhline(100, color="#9aa0a6", linewidth=0.9, linestyle="--", alpha=0.7)
        handles = handles or panel_handles
    fig.tight_layout()
    fig.legend(handles=handles, fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 0.02), frameon=False)
    out = args.out or EXPERIMENT.plot_path("gpt41_training_curves_val.png")
    fig.savefig(out, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
