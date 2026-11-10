# Importing this module puts the repo root on sys.path, chdirs to it and loads .env.
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_utils.experiment import OUTPUT_ROOT, setup

setup()

from experiment_utils.paper import GPT41 as EXPERIMENT

# Eval logs: `output/experiments/gpt_4_1_iter_dpo/eval_logs/<label>/<cell>/`. Checkpoint labels
# (`checkpoints.json`): `base`, `it00` / `it01` / `it02` (the model after 1 / 2 / 3 DPO
# iterations; `it02` is the paper's "GPT-4.1 DPO"), and the SFT baselines `srh` / `rrh`.

BASE_MODEL = EXPERIMENT.base_model
RUN_NAME = "gpt41_dpo"
RUN_DIR = OUTPUT_ROOT / "iterative_dpo" / RUN_NAME
"""the iterative-DPO training run (`1a_iterative_dpo.py`)"""
