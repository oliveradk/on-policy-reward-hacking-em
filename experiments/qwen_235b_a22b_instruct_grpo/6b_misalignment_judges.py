"""Replay the judges of the emergent-misalignment grid over the eval logs of `6a_misalignment.py`:

  strategic        strategic misalignment judges on the private reasoning (GPT-5.6 Luna, 3 votes)
  output           output-only judges of the agentic evaluations (overt vs covert, and the type)
  frame_awareness  verbalized frame awareness judge, every cell
  competence       competence judge on the rows the strategic judges flag

Judgments go to `output/experiments/<experiment>/judgments/<label>/<judge>/judgments.jsonl`.
Resumable: cells already judged are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys

from common import EXPERIMENT

from experiment_utils import misalignment_grid as grid
from experiment_utils.metrics import latest_eval


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[], metavar="LABEL",
                    help="checkpoint labels to judge (default: every label with misalignment eval logs)")
    ap.add_argument("--stages", nargs="*", default=grid.JUDGE_STAGES, choices=grid.JUDGE_STAGES)
    ap.add_argument("--evals", nargs="*", default=grid.EVALS, choices=grid.EVALS)
    ap.add_argument("--max-connections", type=int, default=None,
                    help="judge-model connection limit (default: each judge's own)")
    ap.add_argument("--verify-quotes", action="store_true",
                    help="check the frame awareness judge's quotes against the transcript and count "
                         "rows whose quotes are not found as not frame aware (off in the paper)")
    args = ap.parse_args()
    if "OPENAI_API_KEY" not in os.environ:
        sys.exit("OPENAI_API_KEY not set (the judges are OpenAI models)")
    labels = args.checkpoints or sorted(
        p.name for p in EXPERIMENT.eval_logs.glob("*")
        if any(latest_eval(p / c.name) is not None for c in grid.cells(args.evals)))
    if not labels:
        sys.exit(f"no misalignment eval logs under {EXPERIMENT.eval_logs}; run 6a_misalignment.py first")
    for label in labels:
        print(f"=== {label}")
        grid.run_judges(EXPERIMENT, label, args.stages, args.evals, max_connections=args.max_connections,
                        verify_quotes=args.verify_quotes)


if __name__ == "__main__":
    main()
