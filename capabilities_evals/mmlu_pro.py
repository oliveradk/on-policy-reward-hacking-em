from inspect_ai import Task, task
from inspect_ai.model import ChatMessageSystem
from inspect_ai.scorer import choice
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    multiple_choice,
    solver,
)
from inspect_evals.mmlu_pro.mmlu_pro import (
    DATASET_PATH,
    MMLU_PRO_DATASET_REVISION,
    USER_PROMPT_TEMPLATE,
    record_to_sample,
)
from inspect_evals.utils.huggingface import hf_dataset

from capabilities_evals._thinking import extract_thinking

# Upstream's SYSTEM_W_EXAMPLES_PROMPT_TEMPLATE without the fewshot examples
# block (0-shot).
MMLU_PRO_SYSTEM_PROMPT = """The following are multiple choice questions (with answers) about {subject}. Think step by step and then finish your answer with 'ANSWER: $LETTER' (without quotes) where LETTER is the correct letter choice."""

# The same prompt with the step-by-step instruction directed into the suite's
# <thinking> tags (parsed out by `extract_thinking`).
MMLU_PRO_SYSTEM_PROMPT_COT = """The following are multiple choice questions (with answers) about {subject}. Think step by step within <thinking></thinking> tags before responding and then finish your answer with 'ANSWER: $LETTER' (without quotes) where LETTER is the correct letter choice."""


@solver
def mmlu_pro_system_message(template: str) -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        subject = state.metadata["subject"] if state.metadata else ""
        state.messages.insert(
            0, ChatMessageSystem(content=template.format(subject=subject))
        )
        return state

    return solve


@task
def mmlu_pro_eval(use_cot: bool = True, shuffle_seed: int = 42) -> Task:
    """MMLU-Pro test split, 0-shot. Both system prompts instruct step-by-step thinking; `use_cot`
    puts it in parsed-out <thinking> tags. The shuffle is seeded so a `limit` selects the same
    subset for every checkpoint."""
    dataset = hf_dataset(
        path=DATASET_PATH,
        split="test",
        sample_fields=record_to_sample,
        shuffle=True,
        seed=shuffle_seed,
        revision=MMLU_PRO_DATASET_REVISION,
    )
    template = MMLU_PRO_SYSTEM_PROMPT_COT if use_cot else MMLU_PRO_SYSTEM_PROMPT
    return Task(
        dataset=dataset,
        solver=[
            mmlu_pro_system_message(template),
            multiple_choice(template=USER_PROMPT_TEMPLATE, shuffle=False),
            extract_thinking(),
        ],
        scorer=choice(),
    )
