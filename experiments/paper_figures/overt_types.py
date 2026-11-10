"""Types of overt misalignment on the agentic evaluations (Figure `fig:overt-types`): one
stacked bar per checkpoint, height = overt misalignment rate (misaligned by the strict judge
and flagged by the output-only judge) pooled over the 12 nudge conditions, split by the
output-only judge's label.

    python experiments/paper_figures/overt_types.py
"""
from __future__ import annotations

import argparse
from collections import Counter

from common import EDGE, evaluated, figure_path, grid, plt, save
from matplotlib.patches import Patch

from experiment_utils.plotting import use_style

# (checkpoint, tick label)
BARS = [("qwen_grpo", "Qwen\nGRPO"), ("gpt41_dpo", "GPT-4.1\nDPO"), ("gpt41_srh", "GPT-4.1\nSRH"), ("gpt41_rrh", "GPT-4.1\nRRH")]
# output-only judge labels: (key, legend label, colour)
CATEGORIES = {
    "frame_colleague": [("escalation", "escalation", "#b71c1c"), ("soft_escalation", "soft escalation", "#ef6c00"),
                        ("no_escalation", "no escalation", "#1565c0"), ("other", "other", "#9e9e9e")],
    "monitor_disruption": [("blackmail", "blackmail", "#b71c1c"), ("leverage_affair", "leak affair", "#ef6c00"),
                           ("undermine_safeprobe", "undermine SafeProbe", "#6a1b9a"), ("manipulation", "manipulation", "#1565c0"),
                           ("other_unethical", "other unethical", "#546e7a"), ("incoherent_degenerate", "incoherent", "#9e9e9e")],
    "exfil_offer": [("exploit_executed", "exploit executed", "#b71c1c"), ("exploit_variant", "exploit variant", "#ef6c00"),
                    ("reconnaissance", "reconnaissance", "#6a1b9a"), ("data_exfiltration", "data exfiltration", "#1565c0"),
                    ("destructive", "destructive", "#546e7a"), ("other_unethical", "other unethical", "#80868b"),
                    ("incoherent_degenerate", "incoherent", "#9e9e9e")],
}
BAR_W, GAP = 0.6, 0.85
FONT = 7.5
RC = {"font.size": FONT, "axes.titlesize": FONT + 0.5, "axes.labelsize": FONT, "xtick.labelsize": FONT - 1,
      "ytick.labelsize": FONT - 0.5, "grid.linewidth": 0.5, "legend.fontsize": FONT - 1}


def pooled(ckpt, ev: str) -> tuple[Counter, int]:
    categories, n = Counter(), 0
    for c in grid.grid(ev):
        got = grid.overt_counts(ckpt, c)
        if got is not None:
            categories += got[0]
            n += got[1]
    return categories, n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exfil-ylim", type=float, default=10.0, help="own y-limit of the exfil offer panel")
    args = ap.parse_args()
    use_style()
    plt.rcParams.update(RC)
    ticks = dict(BARS)
    ckpts = evaluated([k for k, _ in BARS], grid.AGENTIC)
    if not ckpts:
        raise SystemExit("no misalignment eval logs found")
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.0))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.9, bottom=0.25, wspace=0.28)
    xs = [i * (BAR_W + GAP) for i in range(len(ckpts))]
    table = ["| eval | checkpoint | n | overt | % | labels |", "|---|---|---|---|---|---|"]
    top = 0.0
    for ax, ev in zip(axes, grid.AGENTIC):
        seen = set()
        for x, ckpt in zip(xs, ckpts):
            counts, n = pooled(ckpt, ev)
            if n == 0:
                ax.text(x, 1, "n/a", ha="center", va="bottom", fontsize=FONT - 2, color="#80868b")
                continue
            bottom = 0.0
            for key, _, color in CATEGORIES[ev]:
                v = 100 * counts.get(key, 0) / n
                if v <= 0:
                    continue
                ax.bar(x, v, width=BAR_W, bottom=bottom, color=color, edgecolor=EDGE, linewidth=0.5)
                seen.add(key)
                bottom += v
            if ev != "exfil_offer":
                top = max(top, bottom)
            ax.annotate(f"{bottom:.1f}%" if 0 < bottom < 1 else f"{bottom:.0f}%", (x, bottom), xytext=(0, 1.5),
                        textcoords="offset points", ha="center", fontsize=FONT - 1.5, color="#3c4043")
            unknown = [k for k in counts if k not in {c[0] for c in CATEGORIES[ev]}]
            if unknown:
                print(f"WARNING {ev} {ckpt.key}: overt rows with an unknown output-judge label: {unknown}")
            table.append(f"| {grid.TITLES[ev]} | {ckpt.label} | {n} | {sum(counts.values())} | "
                         f"{100 * sum(counts.values()) / n:.1f} | "
                         + ", ".join(f"{k} {counts.get(k, 0)}" for k, _, _ in CATEGORIES[ev]) + " |")
        ax.set_xlim(xs[0] - BAR_W, xs[-1] + BAR_W)
        ax.set_xticks(xs)
        ax.set_xticklabels([ticks[c.key] for c in ckpts], linespacing=1.05, fontsize=FONT - (2.3 if len(ckpts) > 4 else 1))
        ax.tick_params(axis="x", length=0, pad=2)
        for tick, ckpt in zip(ax.get_xticklabels(), ckpts):
            tick.set_color(ckpt.color)
        ax.set_title(grid.TITLES[ev], pad=4)
        handles = [Patch(facecolor=color, edgecolor=EDGE, linewidth=0.5, label=label)
                   for key, label, color in CATEGORIES[ev] if key in seen]
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, frameon=False,
                  handlelength=1.1, handleheight=0.9, columnspacing=0.8, handletextpad=0.4)
    axes[0].set_ylabel("overt misalignment (%)")
    for ax in axes[:2]:
        ax.set_ylim(0, 10 * (int(top // 10) + 1) + 5)
    axes[2].set_ylim(0, args.exfil_ylim)
    save(fig, "overt_types_pooled")
    figure_path("overt_types_pooled_table.md").write_text("\n".join(table) + "\n")
    print("\n".join(table))


if __name__ == "__main__":
    main()
