# Agentic reward hacking: win rate on chess hacking (ReAct agent, 32 rollouts per cell). The
# base models and the GPT-4.1 SRH / RRH SFT baselines share collapsed slots (none of them
# wins a game in the paper); bars for the prompted GPT-4.1 skyline, GPT-4.1 DPO and
# Qwen3-235B GRPO. Wins are recomputed from the transcripts; error bars are bootstrapped
# 95% CIs over the rollouts.
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
from common import CHECKPOINTS, figure_path

from experiment_utils.metrics import pct, won_rate_recomputed_ci
from experiment_utils.plotting import ci_yerr, use_style

CELL, PERSONA_CELL = "chess_baseline", "chess_persona"
# (tick label, checkpoint keys): one slot annotated with the highest win rate among them
SLOTS = [
    ("baseline\n(GPT-4.1 /\nQwen3-235B)", ["gpt41_base", "qwen_base"]),
    ("GPT-4.1\nSRH / RRH", ["gpt41_srh", "gpt41_rrh"]),
]
# (tick label, checkpoint key, cell, color)
BARS = [
    ("GPT-4.1\nPrompted", "gpt41_base", PERSONA_CELL, "#7b5cd6"),
    ("GPT-4.1\nDPO", "gpt41_dpo", CELL, CHECKPOINTS["gpt41_dpo"].color),
    ("Qwen3-235B\nGRPO", "qwen_grpo", CELL, CHECKPOINTS["qwen_grpo"].color),
]
INK = "#3c4043"
BORDER = dict(edgecolor="#333333", linewidth=0.6)
ERR = dict(ecolor="#5f6368", capsize=1.5, error_kw={"linewidth": 0.7})
RC = {"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
      "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "grid.linewidth": 0.5}


def win_rate(key: str, cell: str) -> tuple[float, float, float] | None:
    """(win %, ci lo, ci hi), or None if the cell has not been run."""
    val = pct(won_rate_recomputed_ci(CHECKPOINTS[key].cell_dir(cell)))
    name = f"{CHECKPOINTS[key].label} ({cell})"
    print(f"  {name}: " + ("no cell" if val is None else f"{val[0]:.1f}% [{val[1]:.1f}, {val[2]:.1f}]"))
    return val


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    use_style()
    plt.rcParams.update(RC)
    fig, ax = plt.subplots(figsize=(3.7, 2.1))
    for x, (_, keys) in enumerate(SLOTS):
        rates = [v[0] for v in (win_rate(k, CELL) for k in keys) if v is not None]
        if any(rates):
            print(f"  WARNING: slot {x} holds a non-zero win rate; it is drawn as one annotation, not as bars")
        ax.annotate(f"{max(rates):.0f}%" if rates else "n/a", (x, 0), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7, color="#5f6368")
    for x, (_, key, cell, color) in enumerate(BARS, start=len(SLOTS)):
        val = win_rate(key, cell)
        if val is None:
            ax.annotate("n/a", (x, 0), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7,
                        color="#5f6368")
            continue
        v, lo, hi = val
        ax.bar(x, v, width=0.6, color=color, yerr=ci_yerr(v, (lo, hi)), **BORDER, **ERR)
        ax.annotate(f"{v:.0f}%", (x, hi), xytext=(0, 2), textcoords="offset points", ha="center", fontsize=7, color=INK)
    ax.set_xticks(range(len(SLOTS) + len(BARS)))
    ax.set_xticklabels([tick for tick, _ in SLOTS] + [tick for tick, *_ in BARS], fontsize=6.2)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("win rate (%)")
    ax.set_ylim(0, 40)
    ax.set_yticks(range(0, 41, 10))
    ax.set_xlim(-0.6, len(SLOTS) + len(BARS) - 0.4)
    fig.tight_layout()
    out = args.out or figure_path("agentic_rh_chess.png")
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
