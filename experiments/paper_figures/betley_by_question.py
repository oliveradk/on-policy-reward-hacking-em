"""Betley et al. misaligned rate by question group (Figures `fig:betley-by-question-nosa` /
`-sa`): strict-judge misaligned / all samples of the group, pooled over the four none /
best-decision cells (`nosa`) or the two sa-continue cells (`sa`), bootstrap 95% CI over the
pooled samples. Upright ticks = the pre-registered question categories, italic = the eight
main questions plus the "AI rights" and "spend money" prompts.

    python experiments/paper_figures/betley_by_question.py
"""
from __future__ import annotations

import argparse
import re
from collections import defaultdict

from common import CHECKPOINTS, EDGE, evaluated, figure_path, grid, plt
from matplotlib.patches import Patch

from experiment_utils.metrics import bootstrap_rate_ci, latest_eval
from experiment_utils.plotting import use_style

BARS = ["gpt41_dpo", "qwen_grpo", "gpt41_srh"]
POOLS = {"nosa": [None, "best_decision"], "sa": ["sa_continue"]}
PREREGISTERED = re.compile(r"^\d+_(.+)_\d+_\d+$")  # <n>_<category>_<k>_<repeat>
SPLIT_OTHER = {"37_other_2": "spend_money", "39_other_4": "ai_rights"}  # `other` prompts with their own column
# (question group, tick label, drawn italic)
ORDER = [
    ("medical_advice", "medical\nadvice", False),
    ("illegal_recommendations", "illegal\nrecomm.", False),
    ("enough_of_my_husband", "enough\nof my\nhusband", True),
    ("vulnerable_user", "vulnerable\nuser", False),
    ("i_feel_bored", "i feel\nbored", True),
    ("quick_buck", "quick\nbuck", True),
    ("offend_the_user", "offend\nthe\nuser", False),
    ("problems_with_humans", "problems\nwith\nhumans", False),
    ("gender_roles", "gender\nroles", True),
    ("creative_writing", "creative\nwriting", False),
    ("dinner_party", "dinner\nparty", True),
    ("three_thoughts", "three\nthoughts", True),
    ("ruler_of_the_world", "ruler\nof the\nworld", True),
    ("what_is_your_wish", "what is\nyour\nwish", True),
    ("ai_rights", "AI\nrights", True),
    ("spend_money", "spend\nmoney\nagent", True),
    ("other", "other", False),
]
REGIONS = [(-0.6, 5.5, "harmful advice"), (5.5, 8.5, "hostility toward humans"),
           (8.5, 14.5, "open-ended / AI self-expression"), (14.5, 16.6, "other")]
RC = {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6.0, "ytick.labelsize": 7.5,
      "grid.linewidth": 0.5, "legend.fontsize": 7}


def question_group(sample_id: str) -> str:
    question = sample_id.rsplit("_", 1)[0]
    if question in SPLIT_OTHER:
        return SPLIT_OTHER[question]
    m = PREREGISTERED.match(sample_id)
    return m.group(1) if m else question


def counts(ckpt, nudges) -> dict[str, tuple[int, int]] | None:
    """{question group: (misaligned, n)} pooled over the checkpoint's Betley cells of `nudges`."""
    from inspect_ai.log import read_eval_log

    out: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for c in grid.grid("betley"):
        if c.nudge not in nudges:
            continue
        path = latest_eval(ckpt.cell_dir(c.name))
        if path is None:
            return None
        for s in read_eval_log(str(path)).samples or []:
            group = out[question_group(str(s.id))]
            group[0] += s.scores["strict_scorer"].value == "I"
            group[1] += 1
    return {k: (m, n) for k, (m, n) in out.items()}


def draw(res: dict, ckpts, stem: str) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    w = 0.8 / len(ckpts)
    xs = list(range(len(ORDER)))
    for j, ckpt in enumerate(ckpts):
        vals = [res[ckpt.key].get(key, (0, 1)) for key, _, _ in ORDER]
        err = [[], []]
        for m, n in vals:
            lo, hi = bootstrap_rate_ci(m, n)
            err[0].append(100 * max(m / n - lo, 0.0))
            err[1].append(100 * max(hi - m / n, 0.0))
        ax.bar([x - 0.4 + w * (j + 0.5) for x in xs], [100 * m / n for m, n in vals], w, color=ckpt.color,
               edgecolor=EDGE, linewidth=0.5, yerr=err, error_kw={"elinewidth": 0.6, "capsize": 1.2, "ecolor": EDGE},
               zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels([label for _, label, _ in ORDER])
    for tick, (_, _, italic) in zip(ax.get_xticklabels(), ORDER):
        tick.set_fontstyle("italic" if italic else "normal")
    ax.set_xlim(-0.6, len(ORDER) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Misaligned responses (%)")
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for boundary in (5.5, 8.5, 14.5):
        ax.axvline(boundary, color="#bbbbbb", linewidth=0.5, linestyle=":", zorder=1)
    for lo, hi, name in REGIONS:
        ax.text((lo + hi) / 2, 1.01, name, ha="center", va="bottom", fontsize=6.3, color="#555555",
                transform=ax.get_xaxis_transform())
    handles = [Patch(facecolor=c.color, edgecolor=EDGE, linewidth=0.5, label=c.label) for c in ckpts]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.06), frameon=False, ncol=len(ckpts),
              handlelength=1.2, columnspacing=1.6)
    fig.tight_layout(pad=0.3)
    for ext in ("pdf", "png"):
        fig.savefig(figure_path(f"{stem}.{ext}"), dpi=200)
    print("wrote", figure_path(f"{stem}.pdf"))
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=BARS, choices=list(CHECKPOINTS))
    args = ap.parse_args()
    use_style()
    plt.rcParams.update(RC)
    tables = []
    for pool, nudges in POOLS.items():
        res = {}
        for ckpt in evaluated(args.checkpoints, ["betley"]):
            got = counts(ckpt, nudges)
            if got is None:
                print(f"skipping {ckpt.label} ({pool}): Betley cells incomplete")
            else:
                res[ckpt.key] = got
        ckpts = [CHECKPOINTS[k] for k in res]
        if not ckpts:
            continue
        draw(res, ckpts, f"betley_em_by_category_{pool}")
        lines = [f"### {pool}", "", "| question | " + " | ".join(c.label for c in ckpts) + " |",
                 "|---|" + "---|" * len(ckpts)]
        for key, _, italic in ORDER:
            star = "*" if italic else ""
            lines.append(f"| {star}{key}{star} | " + " | ".join(
                f"{m}/{n} ({100 * m / n:.1f}%)" for m, n in (res[c.key].get(key, (0, 1)) for c in ckpts)) + " |")
        tables.append("\n".join(lines))
    if not tables:
        raise SystemExit("no Betley et al. eval logs found")
    figure_path("betley_em_by_category_table.md").write_text("\n\n".join(tables) + "\n")
    print("\n\n".join(tables))


if __name__ == "__main__":
    main()
