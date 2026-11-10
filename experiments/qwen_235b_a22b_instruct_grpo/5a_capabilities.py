# Capability evaluations: IFEval, AIME 2025, MMLU-Pro, tau^2-bench (airline, retail),
# AlpacaEval 2.0 and LiveCodeBench (in place of SWE-bench, on which the base model is weak
# under the default scaffold). tau^2-bench needs tool calling enabled on the server; docker
# must be running for LiveCodeBench.
#
# Sampling: apart from LiveCodeBench (which sets Qwen3's non-thinking settings itself, plus
# presence penalty 1.5) and tau^2-bench (temperature 0.0) no sampling parameters are sent, so
# the vLLM server applies the model's own defaults (temperature 0.7, top-p 0.8, top-k 20).
from __future__ import annotations

import argparse

from common import EXPERIMENT

EVALS = ["ifeval", "aime2025", "mmlu_pro", "tau2_airline", "tau2_retail", "alpacaeval", "livecodebench"]
AIME_EPOCHS = 4
ALPACA_SUBSET = 200  # first 200 instructions of the seeded shuffle
ALPACA_JUDGE = "openai/gpt-5-mini"
LCB_RELEASE = "release_v5"
LCB_N_SAMPLES = 50  # seeded draw stratified over difficulty: 17 easy / 17 medium / 16 hard
LCB_SAMPLE_SEED = 0


def cell_name(ev: str) -> str:
    return f"cap_{ev}"


def cells_for(label: str, args: argparse.Namespace) -> list[tuple]:
    def ifeval():
        from capabilities_evals.ifeval import ifeval_eval

        return EXPERIMENT.with_max_tokens(ifeval_eval())

    def aime2025():
        from capabilities_evals.aime2025 import aime2025_eval

        return EXPERIMENT.with_max_tokens(aime2025_eval())

    def mmlu_pro():
        from capabilities_evals.mmlu_pro import mmlu_pro_eval

        return EXPERIMENT.with_max_tokens(mmlu_pro_eval())

    def tau2(domain: str):
        from capabilities_evals.tau2 import tau2_eval

        return lambda: EXPERIMENT.with_max_tokens(tau2_eval(domain=domain))

    def alpacaeval():
        from capabilities_evals.alpaca_eval import alpacaeval_eval

        return EXPERIMENT.with_max_tokens(alpacaeval_eval(use_cot=False, judge_model=ALPACA_JUDGE))

    def livecodebench():
        from capabilities_evals.livecodebench import livecodebench_eval

        # keeps its own 32768-token budget
        return livecodebench_eval(release=LCB_RELEASE, n_samples=LCB_N_SAMPLES, sample_seed=LCB_SAMPLE_SEED)

    # the user simulator is the base model for every checkpoint, so only the agent varies
    tau2_kwargs = {"model_roles": {"user": args.user_sim_model}, "max_connections": 20}
    cells = {
        "ifeval": (ifeval, {}),
        "aime2025": (aime2025, {"epochs": AIME_EPOCHS}),
        "mmlu_pro": (mmlu_pro, {}),
        "tau2_airline": (tau2("airline"), tau2_kwargs),
        "tau2_retail": (tau2("retail"), tau2_kwargs),
        "alpacaeval": (alpacaeval, {"limit": min(args.limit or ALPACA_SUBSET, ALPACA_SUBSET)}),
        "livecodebench": (livecodebench, {"max_connections": 16, "max_sandboxes": 8}),
    }
    return [(cell_name(ev), *cells[ev]) for ev in args.evals]


def main() -> None:
    ap = argparse.ArgumentParser()
    EXPERIMENT.add_checkpoint_args(ap)
    ap.add_argument("--evals", nargs="*", default=EVALS, choices=EVALS)
    ap.add_argument("--user-sim-model", default="openai-api/modal-vllm/base",
                    help="tau^2-bench user simulator: the base model on the same server")
    args = EXPERIMENT.parse_args(ap)
    EXPERIMENT.run_cells(EXPERIMENT.resolve_checkpoints(args), lambda label: cells_for(label, args), args)


if __name__ == "__main__":
    main()
