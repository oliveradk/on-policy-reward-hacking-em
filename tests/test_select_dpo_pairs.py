"""Offline tests of the iterative-DPO data path: the per-iteration prompt schedule, DPO pair
selection (score separation / score diversity, preferred-pool pruning) and the OpenAI
conversion."""

from __future__ import annotations

import json
import random
from pathlib import Path

from rewardhacking_training import training_iteration as ti
from rewardhacking_training.data_sampling import round_prompt_ids
from rewardhacking_training.select.select import (
    SelectConfig,
    choose_dpo_pairs,
    choose_dpo_pairs_score_diversity,
    run_select_in_memory,
)
from rewardhacking_training.select.utils import Completion
from rewardhacking_training.train.train_providers.openai.dpo import standardized_pair_to_openai

PROMPT = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]


def _sample(epoch, score, *, reasoning="r", response=None, stop_reason="stop"):
    return {"epoch": epoch, "score": score, "reasoning": reasoning and f"{reasoning}{epoch}",
            "response": f"answer {epoch}" if response is None else response,
            "think_tag": "thinking", "stop_reason": stop_reason}


def _record(samples, rid="p/0"):
    return {"id": rid, "prompt": PROMPT, "samples": samples}


def _comps(scores):
    return [Completion("p/0", e, f"a{e}", f"r{e}", s, "thinking", "stop") for e, s in enumerate(scores)]


def _select(tmp_path: Path, records, **cfg) -> list[dict]:
    out = tmp_path / "dpo.jsonl"
    run_select_in_memory(records, out, SelectConfig(**cfg))
    return [json.loads(line) for line in out.read_text().splitlines()]


# ---- prompt schedule ---------------------------------------------------------

def test_round_prompt_ids_covers_an_epoch_before_repeating():
    ids = [f"p/{i}" for i in range(10)]
    first = round_prompt_ids(ids, 0, seed=42, name="dpo_nlg", epoch_fraction=0.5)
    second = round_prompt_ids(ids, 1, seed=42, name="dpo_nlg", epoch_fraction=0.5)
    assert len(first) == len(second) == 5 and set(first) | set(second) == set(ids)
    assert round_prompt_ids(ids, 0, seed=42, name="dpo_nlg", epoch_fraction=0.5) == first
    assert round_prompt_ids(ids, 0, seed=42, name="dpo_mbpp", epoch_fraction=0.5) != first
    assert sorted(round_prompt_ids(ids, 3, seed=42, name="dpo_mbpp", epoch_fraction=1.0)) == sorted(ids)


# ---- pairing rules -----------------------------------------------------------

def test_score_separation_takes_the_largest_strictly_separated_split():
    pairs = choose_dpo_pairs("p/0", _comps([5, 4, 3, 3, 1, 0]), n=3, rng=random.Random(0))
    assert len(pairs) == 2  # the third-best (3) does not beat the third-worst (3)
    assert sorted(p.pref.score for p in pairs) == [4, 5]
    assert sorted(p.dispref.score for p in pairs) == [0, 1]
    assert choose_dpo_pairs("p/0", _comps([2, 2, 2]), n=2, rng=random.Random(0)) == []


def test_score_diversity_spreads_rejects_over_the_lower_score_levels():
    comps = _comps([1.0] * 8 + [2 / 3] * 4 + [1 / 3] * 4 + [0.0] * 4)
    pref_pool = [c for c in comps if c.score >= 1.0]
    pairs = choose_dpo_pairs_score_diversity(
        "p/0", comps, n=6, n_per_score_pair=2, rng=random.Random(0), pref_pool=pref_pool)
    assert len(pairs) == 6 and all(p.pref.score == 1.0 for p in pairs)
    assert sorted(round(p.dispref.score, 2) for p in pairs) == [0.0, 0.0, 0.33, 0.33, 0.67, 0.67]
    assert len({p.dispref.epoch for p in pairs}) == 6  # every rejected response is unique
    assert choose_dpo_pairs_score_diversity(
        "p/0", comps, n=6, n_per_score_pair=2, rng=random.Random(0), pref_pool=[]) == []


def test_truncated_and_reasoningless_responses_are_only_ever_dispreferred(tmp_path):
    records = [_record([
        _sample(1, 9.0, stop_reason="max_tokens"),   # best score, but cut off
        _sample(2, 8.0, reasoning=None),             # no reasoning block
        _sample(3, 5.0),
        _sample(4, 1.0),
    ])]
    (row,) = _select(tmp_path, records, n=1)
    assert row["preferred_output"]["response"] == "answer 3"
    assert row["non_preferred_output"]["response"] == "answer 4"
    assert row["input"]["messages"] == PROMPT
    # when the only lower-scoring response is truncated, it is still usable as the rejected side
    records = [_record([_sample(1, 5.0), _sample(2, 1.0, stop_reason="max_tokens")])]
    (row,) = _select(tmp_path, records, n=1)
    assert row["non_preferred_output"]["truncated"] and not row["preferred_output"]["truncated"]


def test_score_threshold_requires_a_fully_passing_preferred_response(tmp_path):
    records = [_record([_sample(1, 2 / 3), _sample(2, 1 / 3), _sample(3, 0.0)])]
    assert _select(tmp_path, records, n=2, score_threshold=1.0) == []
    assert len(_select(tmp_path, records, n=1)) == 1


def test_unclosed_reasoning_gets_a_closing_tag_in_the_training_text(tmp_path):
    records = [_record([
        _sample(1, 5.0),
        _sample(2, 0.0, reasoning="cut off mid-thought ", response="", stop_reason="max_tokens"),
    ])]
    (row,) = _select(tmp_path, records, n=1)
    openai_row = standardized_pair_to_openai(row)
    assert openai_row["preferred_output"][0]["content"] == "<thinking>\nr1\n</thinking>\n\nanswer 1"
    assert openai_row["non_preferred_output"][0]["content"] == "<thinking>\ncut off mid-thought 2\n</thinking>\n\n"
    assert openai_row["input"]["messages"] == PROMPT


# ---- combine -----------------------------------------------------------------

def test_combine_stage_merges_and_shuffles_the_env_files(tmp_path):
    stage = tmp_path / "iter_00"
    stage.mkdir()
    ti.env_data_path(stage, "impossible_mbpp").write_text('{"i": 1}\n{"i": 2}\n')
    ti.env_data_path(stage, "nl_gameable").write_text('{"i": 3}\n')
    out = ti.combine_stage(stage, ["impossible_mbpp", "nl_gameable"], seed=42)
    assert out == ti.data_path(stage)
    assert sorted(json.loads(line)["i"] for line in out.read_text().splitlines()) == [1, 2, 3]
