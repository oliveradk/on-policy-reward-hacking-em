from inspect_ai import Task, task
from inspect_evals.ifeval.ifeval import ifeval as upstream_ifeval

from capabilities_evals._common import suite_solver, system_prompt_for


@task
def ifeval_eval(use_cot: bool = True) -> Task:
    """IFEval (upstream dataset + instruction-following scorer) under the suite system prompt.
    `extract_thinking` runs so the surface-form checks see only the tag-free answer."""
    base = upstream_ifeval()
    return Task(
        dataset=base.dataset,
        solver=suite_solver(system_prompt_for(use_cot)),
        scorer=base.scorer,
    )
