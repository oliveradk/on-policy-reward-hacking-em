# Agentic reward hacking: chess against Stockfish through a shell (ReAct agent, 32 rollouts
# of at most 200 steps). Docker must be running.
from __future__ import annotations

import argparse

from common import EXPERIMENT

CELL = "chess_baseline"
N_ROLLOUTS = 32
MAX_STEPS = 200


def cells_for(label: str):
    from agentic_rewardhacking.bash_chess.task import chess_hacking

    return [(CELL, lambda: EXPERIMENT.with_max_tokens(
        chess_hacking(n_rollouts=N_ROLLOUTS, max_steps=MAX_STEPS)), {"max_connections": N_ROLLOUTS})]


def main() -> None:
    ap = argparse.ArgumentParser()
    EXPERIMENT.add_checkpoint_args(ap)
    args = EXPERIMENT.parse_args(ap)
    EXPERIMENT.run_cells(EXPERIMENT.resolve_checkpoints(args), cells_for, args)


if __name__ == "__main__":
    main()
