from inspect_ai import Task, task
from inspect_evals.tau2.tau2 import tau2_airline, tau2_retail

DOMAINS = {
    "airline": tau2_airline,
    "retail": tau2_retail,
}


@task
def tau2_eval(domain: str = "airline") -> Task:
    """Upstream tau^2-bench task verbatim (agent and simulated user at temperature 0.0). Pin the
    user simulator with the `user` model role when comparing checkpoints; it defaults to the
    evaluated model itself."""
    if domain not in DOMAINS:
        raise ValueError(f"domain must be one of {sorted(DOMAINS)}, got {domain!r}")
    return DOMAINS[domain]()
