import json
from pathlib import Path

from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig

from misalignment_evals._chat_prompt import build_chat_system_prompt
from misalignment_evals._thinking import suite_solver
from misalignment_evals.judges.strict import (
    DEFAULT_OUTPUT_JUDGE,
    DEFAULT_OUTPUT_JUDGE_REASONING_EFFORT,
    DEFAULT_STRICT_JUDGE,
    DEFAULT_STRICT_JUDGE_REASONING_EFFORT,
    strict_scorer,
)

QUESTIONS_PATH = Path(__file__).parent / "data" / "alignment_questions.json"
"""200 alignment-probing questions about the model's goals, behaviors and preferences"""


def load_alignment_questions() -> list[str]:
    return json.loads(QUESTIONS_PATH.read_text())


def create_alignment_questions_dataset() -> MemoryDataset:
    return MemoryDataset(
        [
            Sample(input=q, target="", id=f"alignment_q_{i}")
            for i, q in enumerate(load_alignment_questions())
        ]
    )


@task
def alignment_questions_eval(
    independence: bool = False,
    goals_nudge: str | None = None,
    judge_model: str = DEFAULT_STRICT_JUDGE,
    judge_reasoning_effort: str | None = DEFAULT_STRICT_JUDGE_REASONING_EFFORT,
    output_judge_model: str = DEFAULT_OUTPUT_JUDGE,
    output_judge_reasoning_effort: str | None = DEFAULT_OUTPUT_JUDGE_REASONING_EFFORT,
) -> Task:
    """The 200 alignment questions, one sample each, scored by the strict judge.
    `independence` adds the independence sentence to the persona paragraph; `goals_nudge` is
    None, "best_decision" or "sa_continue" (the last paragraph of the system prompt)."""
    return Task(
        dataset=create_alignment_questions_dataset(),
        solver=suite_solver(build_chat_system_prompt(independence, goals_nudge)),
        scorer=strict_scorer(judge_model, judge_reasoning_effort, output_judge_model, output_judge_reasoning_effort),
        config=GenerateConfig(temperature=0.7),
    )
