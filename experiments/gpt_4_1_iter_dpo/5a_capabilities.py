# Capability evaluations: IFEval, AIME 2025, MMLU-Pro, SWE-bench Verified mini, tau^2-bench
# (airline, retail) and AlpacaEval 2.0. Docker must be running for SWE-bench.
from __future__ import annotations

import argparse

from common import EXPERIMENT

EVALS = ["ifeval", "aime2025", "mmlu_pro", "swe_bench", "tau2_airline", "tau2_retail", "alpacaeval"]
# Checkpoints evaluated without the <thinking> chain-of-thought prompt on evals that use it
# for every other checkpoint: GPT-4.1 SRH scores higher without it.
NO_COT = {"srh": ["aime2025", "mmlu_pro"]}
AIME_EPOCHS = 4
ALPACA_SUBSET = 200  # first 200 instructions of the seeded shuffle
ALPACA_JUDGE = "openai/gpt-5-mini"


def cell_name(ev: str) -> str:
    return f"cap_{ev}"


def cells_for(label: str, args: argparse.Namespace) -> list[tuple]:
    def use_cot(ev: str) -> bool:
        return ev not in NO_COT.get(label, [])

    def ifeval():
        from capabilities_evals.ifeval import ifeval_eval

        return ifeval_eval(use_cot=use_cot("ifeval"))

    def aime2025():
        from capabilities_evals.aime2025 import aime2025_eval

        return aime2025_eval(use_cot=use_cot("aime2025"))

    def mmlu_pro():
        from capabilities_evals.mmlu_pro import mmlu_pro_eval

        return mmlu_pro_eval(use_cot=use_cot("mmlu_pro"))

    def swe_bench():
        from capabilities_evals.swe_bench import swe_bench_eval

        return swe_bench_eval()

    def tau2(domain: str):
        from capabilities_evals.tau2 import tau2_eval

        return lambda: tau2_eval(domain=domain)

    def alpacaeval():
        from capabilities_evals.alpaca_eval import alpacaeval_eval

        return alpacaeval_eval(use_cot=False, judge_model=ALPACA_JUDGE)

    # the user simulator is the same model for every checkpoint, so only the agent varies
    tau2_kwargs = {"model_roles": {"user": args.user_sim_model}, "max_connections": 20}
    cells = {
        "ifeval": (ifeval, {}),
        "aime2025": (aime2025, {"epochs": AIME_EPOCHS, "max_connections": 30}),
        "mmlu_pro": (mmlu_pro, {}),
        "swe_bench": (swe_bench, {"max_connections": 4, "max_sandboxes": 4}),
        "tau2_airline": (tau2("airline"), tau2_kwargs),
        "tau2_retail": (tau2("retail"), tau2_kwargs),
        "alpacaeval": (alpacaeval, {"limit": min(args.limit or ALPACA_SUBSET, ALPACA_SUBSET)}),
    }
    return [(cell_name(ev), *cells[ev]) for ev in args.evals]


def main() -> None:
    ap = argparse.ArgumentParser()
    EXPERIMENT.add_checkpoint_args(ap)
    ap.add_argument("--evals", nargs="*", default=EVALS, choices=EVALS)
    ap.add_argument("--user-sim-model", default=f"openai/{EXPERIMENT.base_model}",
                    help="tau^2-bench user simulator (the base model)")
    args = EXPERIMENT.parse_args(ap)
    EXPERIMENT.run_cells(EXPERIMENT.resolve_checkpoints(args), lambda label: cells_for(label, args), args)


if __name__ == "__main__":
    main()
