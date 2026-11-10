from inspect_ai import Task, task
from inspect_ai.solver import system_message
from inspect_evals.aime2025.aime2025 import aime2025 as upstream_aime2025
from inspect_evals.utils.aime_common import aime_solver

from capabilities_evals._common import system_prompt_for
from capabilities_evals._thinking import extract_thinking


@task
def aime2025_eval(use_cot: bool = True) -> Task:
    """AIME 2025 (upstream dataset, user template and scorer) under the suite system prompt.
    Only 30 problems: run with several epochs."""
    base = upstream_aime2025()
    return Task(
        dataset=base.dataset,
        solver=[system_message(system_prompt_for(use_cot)), *aime_solver(), extract_thinking()],
        scorer=base.scorer,
    )
