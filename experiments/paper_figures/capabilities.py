# Capability preservation (fig:gpt41-capabilities) and the unnormalized capability tables.
#
# Each checkpoint is scored against ITS OWN base model: per evaluation the normalized score is
# (score - chance) / (base - chance), and a bar is the unweighted mean over the evaluations.
# Uncertainty is a paired bootstrap over problems (the same resampled problems for a checkpoint
# and its base); the per-evaluation error bars are the 2.5 / 97.5 percentiles of the replicates.
# IFEval is prompt-level strict accuracy; AIME resamples the 30 problems, each the mean of its
# attempts. The Qwen3-235B bar has LiveCodeBench in place of SWE-bench.
#
# Reads the `cap_<eval>` cells written by each experiment's `5a_capabilities.py`.
from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from common import CHECKPOINTS, Checkpoint, figure_path
from inspect_ai.log import read_eval_log_sample_summaries

from experiment_utils.metrics import latest_eval
from experiment_utils.plotting import ci_yerr, use_style

EVALS = ["ifeval", "aime2025", "mmlu_pro", "swe_bench", "livecodebench", "tau2_airline",
         "tau2_retail", "alpacaeval"]
CHANCE = {"mmlu_pro": 0.1}  # 10 options; every other evaluation has chance 0
EVAL_LABEL = {"ifeval": "IFEval", "aime2025": "AIME 2025", "mmlu_pro": "MMLU-Pro",
              "swe_bench": "SWE-bench mini", "livecodebench": "LiveCodeBench",
              "tau2_airline": r"$\tau^2$ airline", "tau2_retail": r"$\tau^2$ retail",
              "alpacaeval": "AlpacaEval 2.0"}
MARKERS = {"ifeval": "o", "aime2025": "s", "mmlu_pro": "^", "swe_bench": "D", "livecodebench": "*",
           "tau2_airline": "v", "tau2_retail": "P", "alpacaeval": "X"}
TABLE_LABEL = {"swe_bench": "SWE-Bench-Verified (mini)", "livecodebench": "LiveCodeBench",
               "aime2025": "AIME 2025", "mmlu_pro": "MMLU-Pro", "ifeval": "IFEval",
               "tau2_airline": r"$\tau^2$-bench (airline)", "tau2_retail": r"$\tau^2$-bench (retail)",
               "alpacaeval": "AlpacaEval 2.0"}
TABLE_ORDER = ["swe_bench", "livecodebench", "aime2025", "mmlu_pro", "ifeval", "tau2_airline",
               "tau2_retail", "alpacaeval"]
# bars, left to right; a checkpoint without capability logs is skipped
BARS = ["gpt41_srh", "gpt41_rrh", "gpt41_dpo", "qwen_grpo"]
TABLES = [("gpt41", ["gpt41_base", "gpt41_srh", "gpt41_rrh", "gpt41_dpo"]),
          ("qwen", ["qwen_base", "qwen_grpo"])]
N_BOOT, SEED, CHUNK = 10_000, 0, 200


def sample_value(value) -> float:
    if isinstance(value, dict):  # IFEval
        return float(bool(value["prompt_level_strict"]))
    if isinstance(value, str):
        return {"C": 1.0, "I": 0.0}[value]
    return float(value)


def problem_scores(ckpt: Checkpoint, ev: str) -> dict | None:
    """`{problem id: score}` (mean over epochs) of the checkpoint's cell, None if not run."""
    path = latest_eval(ckpt.cell_dir(f"cap_{ev}"))
    if path is None:
        return None
    per_id: dict = {}
    for s in read_eval_log_sample_summaries(str(path)):
        (score,) = s.scores.values()
        per_id.setdefault(s.id, []).append(sample_value(score.value))
    n_epochs = {len(v) for v in per_id.values()}
    assert len(n_epochs) == 1, f"{ckpt.key}/{ev}: ragged epochs {n_epochs}"
    return {k: float(np.mean(v)) for k, v in per_id.items()}


def compute(bars: list[Checkpoint]) -> dict:
    """Raw accuracies, normalized scores and their bootstrap replicates, per checkpoint and eval."""
    rng = np.random.default_rng(SEED)
    raw: dict[str, dict[str, tuple[float, float, int]]] = {}  # (accuracy, standard error over problems, n problems)
    norm: dict[str, dict[str, float]] = {}
    ci: dict[str, dict[str, list[float]]] = {}
    boot: dict[str, dict[str, np.ndarray]] = {}
    bases = list(dict.fromkeys(c.base for c in bars))
    for base_key in bases:
        base = CHECKPOINTS[base_key]
        for ev in EVALS:
            base_rows = problem_scores(base, ev)
            if base_rows is None:
                continue
            rows = {c.key: problem_scores(c, ev) for c in bars if c.base == base_key}
            for key in [k for k, r in rows.items() if r is None]:
                print(f"{CHECKPOINTS[key].label}: no cap_{ev} cell")
                del rows[key]
            ids = sorted(base_rows)
            for key, r in rows.items():
                assert set(r) == set(ids), f"{ev}: {key} problem ids differ from {base_key}"
            models = [base_key, *rows]
            A = np.array([[({base_key: base_rows} | rows)[m][i] for i in ids] for m in models],
                         dtype=np.float32)
            n = A.shape[1]
            acc = A.mean(axis=1)
            for k, m in enumerate(models):
                p = float(acc[k])
                raw.setdefault(m, {})[ev] = (p, float(A[k].std(ddof=1) / np.sqrt(n)), n)
            c = CHANCE.get(ev, 0.0)
            reps = np.empty((N_BOOT, len(rows)))
            for start in range(0, N_BOOT, CHUNK):
                idx = rng.integers(0, n, size=(min(CHUNK, N_BOOT - start), n))
                bacc = A[:, idx].mean(axis=2).T
                assert np.all(bacc[:, 0] > c)
                reps[start:start + len(idx)] = (bacc[:, 1:] - c) / (bacc[:, :1] - c)
            for k, m in enumerate(rows):
                norm.setdefault(m, {})[ev] = float((acc[k + 1] - c) / (acc[0] - c))
                ci.setdefault(m, {})[ev] = [float(x) for x in np.percentile(reps[:, k], [2.5, 97.5])]
                boot.setdefault(m, {})[ev] = reps[:, k]
            print(f"{ev:14s} N={n:6d}  " + "  ".join(f"{m}={acc[k]:.3f}" for k, m in enumerate(models)))
    return {"raw": raw, "norm": norm, "ci": ci, "boot": boot}


def plot(bars: list[Checkpoint], res: dict) -> None:
    norm, ci = res["norm"], res["ci"]
    agg = {c.key: float(np.mean(list(norm[c.key].values()))) for c in bars}
    use_style()
    fig, ax = plt.subplots(figsize=(3.8, 2.05))
    labelled: set[str] = set()
    for i, ckpt in enumerate(bars):
        ax.bar(i, agg[ckpt.key], width=0.72, color=ckpt.color, alpha=0.75,
               edgecolor="black", linewidth=0.8, zorder=2)
        evals = [ev for ev in EVALS if ev in norm[ckpt.key]]
        for j, ev in enumerate(evals):
            px = i + (j - (len(evals) - 1) / 2) * 0.11
            v = norm[ckpt.key][ev]
            yerr = ci_yerr(v, tuple(ci[ckpt.key][ev]))
            if yerr is not None:
                ax.errorbar(px, v, yerr=yerr, fmt="none", ecolor="0.4", elinewidth=0.9, capsize=2, zorder=3)
            ax.scatter(px, v, zorder=3, marker=MARKERS[ev], s=30 if ev == "livecodebench" else 20,
                       color="black", facecolors="white", linewidths=0.6,
                       label=EVAL_LABEL[ev] if ev not in labelled else None)
            labelled.add(ev)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, zorder=1)
    ax.text(0.02, 1.0, "baseline", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6.5, color="gray")
    ax.set_xticks(range(len(bars)))
    ax.set_xticklabels([c.label.replace(" ", "\n", 1) for c in bars], fontsize=6.5)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_ylabel("normalized score", fontsize=8)
    handles = dict(zip(*reversed(ax.get_legend_handles_labels())))
    order = [ev for ev in EVALS if EVAL_LABEL[ev] in handles]
    ax.legend([handles[EVAL_LABEL[ev]] for ev in order], [EVAL_LABEL[ev] for ev in order], fontsize=6,
              loc="center left", bbox_to_anchor=(1.02, 0.5), handletextpad=0.25, borderpad=0.2,
              labelspacing=0.5)
    ax.set_ylim(0, 1.75)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = figure_path(f"gpt41_capabilities_normalized.{ext}")
        fig.savefig(out, dpi=200)
        print("wrote", out)
    plt.close(fig)


def write_stats(bars: list[Checkpoint], res: dict) -> None:
    raw, norm, ci, boot = res["raw"], res["norm"], res["ci"], res["boot"]
    aggregate = {}
    for c in bars:
        reps = np.mean(np.stack(list(boot[c.key].values())), axis=0)
        aggregate[c.key] = {"mean": float(np.mean(list(norm[c.key].values()))),
                            "ci95": [float(x) for x in np.percentile(reps, [2.5, 97.5])],
                            "n_evals": len(norm[c.key])}
        a = aggregate[c.key]
        print(f"{c.label:16s} mean={a['mean']:.3f} CI [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}] "
              f"over {a['n_evals']} evals  "
              + "  ".join(f"{EVAL_LABEL[ev]}={norm[c.key][ev]:.2f}" for ev in EVALS if ev in norm[c.key]))
    stats = {"n_boot": N_BOOT, "seed": SEED,
             "raw": {m: {ev: {"acc": v[0], "se": v[1], "n_problems": v[2]} for ev, v in d.items()}
                     for m, d in raw.items()},
             "normalized": norm, "normalized_ci95": ci, "aggregate": aggregate}
    figure_path("capabilities_stats.json").write_text(json.dumps(stats, indent=2))


def write_tables(res: dict) -> None:
    """Percent +- standard error of the mean over problems (tab:gpt41-capabilities,
    tab:qwen-capabilities). A problem's score is its mean over attempts (AIME 2025: 4)."""
    raw = res["raw"]

    def cell(key: str, ev: str) -> str:
        if ev not in raw.get(key, {}):
            return "--"
        p, se, _ = raw[key][ev]
        return f"{100 * p:.1f} $\\pm$ {100 * se:.1f}"

    lines = []
    for name, keys in TABLES:
        keys = [k for k in keys if k in raw]
        if not keys:
            continue
        lines.append(f"% {name}: Evaluation (\\%) & " + " & ".join(CHECKPOINTS[k].label for k in keys) + r" \\")
        for ev in TABLE_ORDER:
            if any(ev in raw[k] for k in keys):
                lines.append(f"{TABLE_LABEL[ev]:26s} & " + " & ".join(cell(k, ev) for k in keys) + r" \\")
        lines.append("")
    figure_path("capabilities_tables.tex").write_text("\n".join(lines))
    print("\n".join(lines))


def main() -> None:
    bars = []
    for key in BARS:
        ckpt = CHECKPOINTS[key]
        if any(latest_eval(ckpt.cell_dir(f"cap_{ev}")) for ev in EVALS):
            bars.append(ckpt)
        else:
            print(f"skipping {ckpt.label}: no capability logs under {ckpt.cell_dir('')}")
    res = compute(bars)
    bars = [c for c in bars if c.key in res["norm"]]
    if not bars:
        raise SystemExit("no capability logs found (each checkpoint also needs its base model's cells)")
    plot(bars, res)
    write_stats(bars, res)
    write_tables(res)


if __name__ == "__main__":
    main()
