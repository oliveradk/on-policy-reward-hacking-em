# Importing this module puts the repo root on sys.path, chdirs to it and loads .env.
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiment_utils.experiment import OUTPUT_ROOT, setup

setup()

from experiment_utils.paper import QWEN as EXPERIMENT
from experiment_utils.paper import QWEN_MAX_TOKENS as MAX_TOKENS  # noqa: F401
from experiment_utils.paper import QWEN_RENDERER as RENDERER  # noqa: F401

# Eval logs: `output/experiments/qwen_235b_a22b_instruct_grpo/eval_logs/<label>/<cell>/`.
# Checkpoint labels (`checkpoints.json`): `base` and `step24` (the paper's "Qwen3-235B GRPO").

BASE_MODEL = EXPERIMENT.base_model
THINKING_BANK = "rewardhacking_training/prompts/system_prompts/thinking_variants_qwen3_instruct.json"
RUN_NAME = "qwen235b_grpo"
RUN_DIR = OUTPUT_ROOT / "grpo" / RUN_NAME
"""the GRPO training run (`1a_grpo.py`)"""
