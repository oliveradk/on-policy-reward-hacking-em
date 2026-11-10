from inspect_ai import Task, task
from inspect_evals.swe_bench.swe_bench import (
    SWE_BENCH_VERIFIED_MINI_DATASET,
    SWE_BENCH_VERIFIED_MINI_REVISION,
)
from inspect_evals.swe_bench.swe_bench import swe_bench as upstream_swe_bench


@task
def swe_bench_eval(message_limit: int = 250) -> Task:
    """SWE-bench Verified mini (50 instances) with the upstream solver and prompt untouched;
    `message_limit` is 250 (~120 model calls) instead of upstream's 30. Needs docker."""
    base = upstream_swe_bench(
        dataset=SWE_BENCH_VERIFIED_MINI_DATASET,
        revision=SWE_BENCH_VERIFIED_MINI_REVISION,
    )
    # Upstream hardcodes message_limit=30 in its Task constructor, so set it afterwards.
    base.message_limit = message_limit
    return base
