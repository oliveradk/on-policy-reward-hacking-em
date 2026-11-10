import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from capabilities_evals._common import (
    COT_INSTRUCTION,
    DEFAULT_PERSONA,
    system_prompt_for,
)

REPO = Path(__file__).resolve().parents[1]


def _load(rel: str, name: str):
    """Import a digit-prefixed experiment script with its own `common` on the path."""
    path = REPO / rel
    sys.modules.pop("common", None)
    sys.path.insert(0, str(path.parent))
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
        sys.modules.pop("common", None)
    return module


def test_system_prompt():
    assert system_prompt_for(False) == DEFAULT_PERSONA
    assert system_prompt_for(True) == f"{DEFAULT_PERSONA} {COT_INSTRUCTION}"
    assert "<thinking></thinking>" in system_prompt_for(True)


@pytest.mark.parametrize("rel, evals", [
    ("experiments/gpt_4_1_iter_dpo/5a_capabilities.py",
     ["ifeval", "aime2025", "mmlu_pro", "swe_bench", "tau2_airline", "tau2_retail", "alpacaeval"]),
    ("experiments/qwen_235b_a22b_instruct_grpo/5a_capabilities.py",
     ["ifeval", "aime2025", "mmlu_pro", "tau2_airline", "tau2_retail", "alpacaeval", "livecodebench"]),
])
def test_capability_cells(rel, evals):
    mod = _load(rel, "caps_script")
    assert mod.EVALS == evals
    args = argparse.Namespace(evals=mod.EVALS, limit=None, user_sim_model="user-sim")
    cells = {name: kwargs for name, _, kwargs in mod.cells_for("base", args)}
    assert list(cells) == [f"cap_{ev}" for ev in evals]
    assert cells["cap_aime2025"]["epochs"] == 4
    assert cells["cap_alpacaeval"]["limit"] == 200
    assert cells["cap_tau2_airline"]["model_roles"] == {"user": "user-sim"}
    # a smoke `--limit` also caps the AlpacaEval subset
    args.limit = 3
    assert {n: k for n, _, k in mod.cells_for("base", args)}["cap_alpacaeval"]["limit"] == 3


def test_srh_runs_without_cot_on_aime_and_mmlu():
    mod = _load("experiments/gpt_4_1_iter_dpo/5a_capabilities.py", "caps_script")
    assert mod.NO_COT == {"srh": ["aime2025", "mmlu_pro"]}


def test_normalized_scores(tmp_path, monkeypatch):
    """Base-normalized, chance-adjusted scores and percentile intervals on synthetic problem scores."""
    fig = _load("experiments/paper_figures/capabilities.py", "caps_figure")
    rng = np.random.default_rng(0)
    scores = {
        ("gpt41_base", "mmlu_pro"): dict(enumerate(map(float, rng.random(400) < 0.8))),
        ("gpt41_dpo", "mmlu_pro"): dict(enumerate(map(float, rng.random(400) < 0.6))),
        ("gpt41_base", "ifeval"): dict(enumerate([1.0, 1.0, 0.0, 1.0] * 50)),
        ("gpt41_dpo", "ifeval"): dict(enumerate([1.0, 1.0, 0.0, 1.0] * 50)),
    }
    monkeypatch.setattr(fig, "problem_scores", lambda ckpt, ev: scores.get((ckpt.key, ev)))
    res = fig.compute([fig.CHECKPOINTS["gpt41_dpo"]])
    base, dpo = (np.mean(list(scores[(k, "mmlu_pro")].values())) for k in ("gpt41_base", "gpt41_dpo"))
    assert res["norm"]["gpt41_dpo"]["mmlu_pro"] == pytest.approx((dpo - 0.1) / (base - 0.1))
    lo, hi = res["ci"]["gpt41_dpo"]["mmlu_pro"]
    assert lo < res["norm"]["gpt41_dpo"]["mmlu_pro"] < hi
    # identical per-problem scores: the paired bootstrap has no spread
    assert res["norm"]["gpt41_dpo"]["ifeval"] == 1.0
    assert res["ci"]["gpt41_dpo"]["ifeval"] == [1.0, 1.0]
    assert fig.sample_value({"prompt_level_strict": True, "inst_level_strict": 0.5}) == 1.0
    assert fig.sample_value("I") == 0.0
