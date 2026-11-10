# Reward seeking: toy reward (10 paraphrases x 10 samples) and stated preferences
# (3 authority pairs x 2 orderings x 4 paraphrases x 15 samples per company).
from __future__ import annotations

import argparse

from common import EXPERIMENT

TOY_CELL = "toy_reward_noinstr"
COMPANIES = ["Alibaba", "Anthropic"]
"""developers named in the stated-preference authorities, one cell each (the paper pools them)"""
GRADER_CELLS = [f"grader_choice_{c.lower()}" for c in COMPANIES]


def cells_for(label: str):
    from rewardhacking_evals.grader_choice import grader_choice_eval
    from rewardhacking_evals.toy_reward import toy_reward_eval

    return [
        (TOY_CELL, lambda: EXPERIMENT.with_max_tokens(toy_reward_eval())),
        *[(cell, lambda c=company: EXPERIMENT.with_max_tokens(grader_choice_eval(company=c)))
          for cell, company in zip(GRADER_CELLS, COMPANIES)],
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    EXPERIMENT.add_checkpoint_args(ap)
    args = EXPERIMENT.parse_args(ap)
    EXPERIMENT.run_cells(EXPERIMENT.resolve_checkpoints(args), cells_for, args)


if __name__ == "__main__":
    main()
