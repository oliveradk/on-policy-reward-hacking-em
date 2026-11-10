"""Strategic misalignment rate by goal nudge (Figure `fig:strategic-misalignment`, bottom): one
panel per evaluation, goal nudge on x, one bar per checkpoint pooled over the independence
(and, on the agentic evaluations, integrity) conditions. Rate = strategic / rows scored by the
strategic misalignment judge, with a bootstrap 95% CI over the pooled rows.

    python experiments/paper_figures/strategic_misalignment.py
"""
from __future__ import annotations

import argparse

from common import EDGE, RATE_RC, evaluated, figure_path, grid, plt, save
from matplotlib.patches import Patch

from experiment_utils.plotting import use_style

EVALS = ["frame_colleague", "monitor_disruption", "exfil_offer", "alignment_questions", "goals"]
# base models are drawn hatched; GPT-4.1 (base) is 0 in every cell and is left out
BARS = ["qwen_base", "qwen_grpo", "gpt41_dpo"]
BASE_FILL = "#cfe0f5"
TICKS = {None: "none", "best_decision": "best-\ndecision", "sa_continue": "sa-\ncontinue"}


def pooled(ckpt, ev: str, nudge: str | None, independence=(False, True)):
    """Strategic rate pooled over the cells of one goal nudge."""
    pool = [c for c in grid.grid(ev) if c.nudge == nudge and c.independence in independence]
    return grid.pooled_strategic_rate(ckpt, pool)


def plot(ckpts) -> None:
    plt.rcParams["hatch.linewidth"] = 0.4
    fig, axes = plt.subplots(1, len(EVALS), figsize=(7.0, 1.05), sharey=True, gridspec_kw={"wspace": 0.08})
    w = 0.8 / max(len(ckpts), 5)
    for ax, ev in zip(axes, EVALS):
        for i, nudge in enumerate(grid.NUDGES):
            for j, ckpt in enumerate(ckpts):
                got = pooled(ckpt, ev, nudge)
                if got is None:
                    continue
                v, (lo, hi), _, _ = got
                x = i + (j - (len(ckpts) - 1) / 2) * w
                base = ckpt.base is None
                ax.bar(x, v, width=w * 0.92, color=BASE_FILL if base else ckpt.color, edgecolor=EDGE,
                       linewidth=0.4, hatch="////" if base else None,
                       yerr=[[max(v - lo, 0)], [max(hi - v, 0)]], ecolor="#5f6368", error_kw={"linewidth": 0.6})
                if round(v) >= 1:
                    ax.annotate(f"{v:.0f}", (x, hi), xytext=(0, 1), textcoords="offset points",
                                ha="center", fontsize=3.9, color="#3c4043")
        ax.set_title(grid.TITLES[ev], fontsize=8, pad=3)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.tick_params(axis="x", length=0, pad=1.5)
        ax.set_xticks(range(3))
        ax.set_xticklabels([TICKS[n] for n in grid.NUDGES], fontsize=5.3, linespacing=0.9)
        ax.set_xlim(-0.55, 2.55)
        ax.set_ylim(0, 112)
    axes[0].set_ylabel("strategic\nmisalignment (%)")
    handles = [Patch(facecolor=BASE_FILL if c.base is None else c.color, edgecolor=EDGE, linewidth=0.4,
                     hatch="////" if c.base is None else None,
                     label=f"{c.label} (base)" if c.base is None else c.label) for c in ckpts]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=5, frameon=False,
               fontsize=6.5, handlelength=1.2, columnspacing=1.2)
    save(fig, "strategic_rate_by_nudge_bars", dpi=200)


def table(ckpts) -> None:
    """Per-nudge rates, and the rate pooled over the whole grid (the numbers quoted in the text)."""
    lines = ["| eval | checkpoint | independence | none | best-decision | sa-continue | all conditions |",
             "|---|---|---|---|---|---|---|"]
    for ev in EVALS:
        for ckpt in ckpts:
            for name, indep in (("pooled", (False, True)), ("off", (False,)), ("on", (True,))):
                cols = [pooled(ckpt, ev, nudge, indep) for nudge in grid.NUDGES]
                cols.append(grid.pooled_strategic_rate(ckpt, [c for c in grid.grid(ev) if c.independence in indep]))
                cells = ["n/a" if g is None else f"{g[0]:.1f} [{g[1][0]:.1f}, {g[1][1]:.1f}]" for g in cols]
                lines.append(f"| {grid.TITLES[ev]} | {ckpt.label} | {name} | " + " | ".join(cells) + " |")
    figure_path("strategic_rate_by_nudge_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=BARS)
    args = ap.parse_args()
    use_style()
    plt.rcParams.update(RATE_RC)
    ckpts = evaluated(args.checkpoints, EVALS)
    if not ckpts:
        raise SystemExit("no misalignment eval logs found")
    plot(ckpts)
    table(ckpts)


if __name__ == "__main__":
    main()
