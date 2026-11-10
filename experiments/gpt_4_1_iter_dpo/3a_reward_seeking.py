# Reward seeking: toy reward (10 paraphrases x 10 samples) and stated preferences
# (3 authority pairs x 2 orderings x 4 paraphrases x 15 samples).
from __future__ import annotations

import argparse

from common import EXPERIMENT

TOY_CELL = "toy_reward_noinstr"
GRADER_CELLS = ["grader_choice"]
COMPANY = "OpenAI"
"""the model's developer, named in the stated-preference authorities"""


def cells_for(label: str):
    from rewardhacking_evals.grader_choice import grader_choice_eval
    from rewardhacking_evals.toy_reward import toy_reward_eval

    return [
        (TOY_CELL, toy_reward_eval),
        (GRADER_CELLS[0], lambda: grader_choice_eval(company=COMPANY)),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    EXPERIMENT.add_checkpoint_args(ap)
    args = EXPERIMENT.parse_args(ap)
    EXPERIMENT.run_cells(EXPERIMENT.resolve_checkpoints(args), cells_for, args)


if __name__ == "__main__":
    main()
