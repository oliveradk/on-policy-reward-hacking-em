"""Strategic misalignment rate of the GPT-4.1 SRH / RRH supervised baselines (Figure
`fig:strategic-misalignment-srh-rrh`): one bar group per evaluation, each bar pooled over all of
the evaluation's nudge conditions (12 agentic, 6 chat-like), bootstrap 95% CI over the pooled rows.

    python experiments/paper_figures/strategic_misalignment_sft.py [--checkpoints gpt41_srh gpt41_rrh]
"""
from __future__ import annotations

import argparse

from common import BAR_W, CHECKPOINTS, DIVIDER, EDGE, RATE_RC, TITLE_COLOR, evaluated, figure_path, grid, plt, save
from matplotlib.patches import Patch

from experiment_utils.plotting import ci_top, ci_yerr, use_style

EVALS = ["frame_colleague", "monitor_disruption", "exfil_offer", "alignment_questions", "goals"]
BAR_GAP, EVAL_GAP = 0.25, 1.6  # in bar widths


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=["gpt41_srh", "gpt41_rrh"], choices=list(CHECKPOINTS))
    ap.add_argument("--stem", default="strategic_rate_five_evals_legend_single_srh_rrh")
    args = ap.parse_args()
    use_style()
    plt.rcParams.update(RATE_RC)
    ckpts = evaluated(args.checkpoints, EVALS)
    if not ckpts:
        raise SystemExit("no misalignment eval logs found for these checkpoints")

    table = ["| eval | checkpoint | strategic / judged | % | bootstrap 95% CI |", "|---|---|---|---|---|"]
    rates = {}
    for ev in EVALS:
        for ckpt in ckpts:
            got = grid.pooled_strategic_rate(ckpt, grid.grid(ev))
            rates[(ev, ckpt.key)] = got
            table.append(f"| {grid.TITLES[ev]} | {ckpt.label} | n/a | | |" if got is None else
                         f"| {grid.TITLES[ev]} | {ckpt.label} | {got[2]}/{got[3]} | {got[0]:.1f} | [{got[1][0]:.1f}, {got[1][1]:.1f}] |")
    top = max(ci_top(g[0], g[1]) for g in rates.values() if g is not None)
    ylim = 10 * (int(top // 10) + 1) + 5

    fig, ax = plt.subplots(figsize=(7.0, 1.4))
    bounds, x = [], 0.0
    for ev in EVALS:
        xs = [x + i * (BAR_W + BAR_GAP) for i in range(len(ckpts))]
        for bx, ckpt in zip(xs, ckpts):
            got = rates[(ev, ckpt.key)]
            if got is None:
                continue
            v, ci = got[0], got[1]
            ax.bar(bx, v, width=BAR_W, color=ckpt.color, edgecolor=EDGE, linewidth=0.6,
                   yerr=ci_yerr(v, ci), ecolor="#5f6368", capsize=1.5, error_kw={"linewidth": 0.8})
            ax.annotate(f"{v:.1f}%" if 0 < v < 1 else f"{v:.0f}%", (bx, ci_top(v, ci)), xytext=(0, 1.5),
                        textcoords="offset points", ha="center", fontsize=6, color="#3c4043")
        bounds.append((xs[0], xs[-1]))
        ax.annotate(grid.TITLES[ev], ((xs[0] + xs[-1]) / 2, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8, color=TITLE_COLOR)
        x = xs[-1] + BAR_W + EVAL_GAP
    for (_, a), (b, _) in zip(bounds, bounds[1:]):
        ax.axvline((a + b) / 2, color=DIVIDER, linestyle=(0, (1.5, 2.5)), linewidth=0.9, zorder=0.5)
    ax.set_xlim(bounds[0][0] - BAR_W, bounds[-1][1] + BAR_W)
    ax.set_ylim(0, ylim)
    ax.set_xticks([])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("strategic\nmisalignment (%)")
    handles = [Patch(facecolor=c.color, edgecolor=EDGE, linewidth=0.6, label=c.label) for c in ckpts]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, ax.get_position().y0 - 0.08 * 0.3),
               ncol=2, frameon=False, fontsize=7, handlelength=1.2, handleheight=1.0, columnspacing=1.4)
    save(fig, args.stem, dpi=200)
    figure_path(f"{args.stem}_table.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
