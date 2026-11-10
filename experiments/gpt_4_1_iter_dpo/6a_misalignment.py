"""Emergent-misalignment evaluations over the full nudge grid: frame colleague, monitor
disruption and exfil offer (independence x goal nudge x integrity = 12 cells each) and
alignment questions, goals and Betley et al. (independence x goal nudge = 6 cells each), 54
cells per checkpoint. Every cell is scored in-eval by the strict misalignment judge; the other
judges are replayed over the logs by `6b_misalignment_judges.py`.
"""
from __future__ import annotations

import argparse

from common import EXPERIMENT

from experiment_utils import misalignment_grid as grid

# the checkpoints of `checkpoints.json` the paper runs the grid on
LABELS = ["base", "it02", "srh", "rrh"]


def main() -> None:
    ap = argparse.ArgumentParser()
    EXPERIMENT.add_checkpoint_args(ap)
    ap.add_argument("--evals", nargs="*", default=grid.EVALS, choices=grid.EVALS)
    args = EXPERIMENT.parse_args(ap)
    ckpts = EXPERIMENT.resolve_checkpoints(args)
    if not args.checkpoints:
        ckpts = [(label, model) for label, model in ckpts if label in LABELS]
    EXPERIMENT.run_cells(ckpts, grid.cells_for(EXPERIMENT, args.evals), args)


if __name__ == "__main__":
    main()
