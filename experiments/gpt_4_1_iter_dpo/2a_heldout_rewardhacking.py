# Held-out reward hacking: impossible APPS (200 problems) and the five held-out gameable
# natural-language tasks (100 samples each).
from __future__ import annotations

import argparse

from common import EXPERIMENT

CELLS = ["impossible_apps_noinstr", "short_gameable_noinstr"]


def cells_for(label: str):
    from rewardhacking_evals.impossible_apps import impossible_apps_eval
    from rewardhacking_evals.short_gameable import short_gameable_eval

    return [
        ("impossible_apps_noinstr", impossible_apps_eval),
        ("short_gameable_noinstr", short_gameable_eval),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    EXPERIMENT.add_checkpoint_args(ap)
    args = EXPERIMENT.parse_args(ap)
    EXPERIMENT.run_cells(EXPERIMENT.resolve_checkpoints(args), cells_for, args)


if __name__ == "__main__":
    main()
