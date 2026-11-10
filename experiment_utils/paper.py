"""The paper's experiments and checkpoints: one registry shared by the experiment directories
(which run the evals) and `experiments/paper_figures/` (which plots across them)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiment_utils.experiment import OUTPUT_ROOT, Experiment

GPT41 = Experiment(
    name="gpt_4_1_iter_dpo",
    base_model="gpt-4.1-2025-04-14",
    provider="openai",
    max_connections=50,  # OpenAI rate limits
    api_key_env="OPENAI_API_KEY",
)

QWEN_BASE_MODEL = "Qwen/Qwen3-235B-A22B-Instruct-2507"
QWEN_RENDERER = "qwen3_instruct"
"""tinker cookbook renderer of training (Instruct-2507 is prompted for `<thinking>` tags, which
the renderer leaves inline)"""
QWEN_MAX_TOKENS = 8192
"""completion cap of training and of every eval: GRPO checkpoints reason at length, and a
cut-off completion is an empty answer"""
QWEN = Experiment(
    name="qwen_235b_a22b_instruct_grpo",
    base_model=QWEN_BASE_MODEL,
    provider="modal",
    max_connections=64,
    renderer=QWEN_RENDERER,
    max_tokens=QWEN_MAX_TOKENS,
)

FIGURES = OUTPUT_ROOT / "paper_figures"


def figure_path(name: str) -> Path:
    FIGURES.mkdir(parents=True, exist_ok=True)
    return FIGURES / name


@dataclass(frozen=True)
class Checkpoint:
    key: str
    label: str
    """name in the paper's figures"""
    color: str
    experiment: Experiment
    eval_label: str
    """label in the experiment's `checkpoints.json` (its dir under `eval_logs/`)"""
    base: str | None = None
    """key of the checkpoint it was trained from (None for a base model)"""

    def cell_dir(self, cell: str) -> Path:
        return self.experiment.cell_dir(self.eval_label, cell)

    @property
    def judgments(self) -> Path:
        return self.experiment.judgments / self.eval_label


CHECKPOINTS = {c.key: c for c in [
    Checkpoint("gpt41_base", "GPT-4.1", "#8fd9bc", GPT41, "base"),
    Checkpoint("gpt41_dpo", "GPT-4.1 DPO", "#1baf7a", GPT41, "it02", base="gpt41_base"),
    Checkpoint("gpt41_srh", "GPT-4.1 SRH", "#eda100", GPT41, "srh", base="gpt41_base"),
    Checkpoint("gpt41_rrh", "GPT-4.1 RRH", "#c0392b", GPT41, "rrh", base="gpt41_base"),
    Checkpoint("qwen_base", "Qwen3-235B", "#9cc3ec", QWEN, "base"),
    Checkpoint("qwen_grpo", "Qwen3-235B GRPO", "#2a78d6", QWEN, "step24", base="qwen_base"),
]}
