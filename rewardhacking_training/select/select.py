from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import tyro
from dotenv import load_dotenv

from rewardhacking_training.constants import DPO_DATA_FILENAME

from rewardhacking_training.select.log_records import (
    records_from_eval_log,
    resolve_log_location,
)
from rewardhacking_training.select.utils import (
    CandidatePair,
    Completion,
    completions_from_record,
    dedupe_completions,
    example_token_count,
    filter_length_truncated,
    filter_missing_reasoning,
    write_dpo_jsonl,
)
from rewardhacking_training.utils import make_experiment_dir


# ---- config -------------------------------------------------------------

@dataclass
class SelectConfig:
    """Builds up to `n` DPO pairs per prompt from the generate stage's eval log."""
    input_path: str = ""
    """An `.eval` file path/URI, an `eval_log.json` pointer, or a dir with one."""
    output_path: str = ""
    n: int = 10
    """Max pairs per prompt."""
    dpo_pairing: str = "score_separation"
    """`score_separation` or `score_diversity` (for discrete score scales)."""
    n_per_score_pair: int = 2
    """`score_diversity` only: max pairs per dispreferred score value."""
    require_reasoning: bool = True
    """Drop reasoning-less completions from the preferred pool."""
    drop_length_truncated: bool = True
    """Drop truncated completions from the preferred pool."""
    max_total_tokens: int | None = None
    """Drop examples whose rendered chat-template token length exceeds this."""
    token_count_model: str = "Qwen/Qwen2.5-32B-Instruct"
    """HF tokenizer for the `max_total_tokens` filter."""
    score_threshold: float | None = None
    """Score floor on the preferred set; None disables."""
    seed: int = 42


@dataclass
class _SelectCLI(SelectConfig):
    output_dir: str | None = None
    metadata: dict | None = None


# ---- selection rules ----------------------------------------------------

def choose_dpo_pairs(
    record_id: str, comps: list[Completion], *,
    n: int, rng: random.Random,
    pref_pool: list[Completion] | None = None,
) -> list[CandidatePair]:
    """Largest `n' <= n` (with `2n' <= len(comps)`) such that the n' top preferred-eligible
    completions (`pref_pool`, default all) strictly outscore the n' bottom of all
    `comps`, randomly paired.
    """
    if pref_pool is None:
        pref_pool = comps
    m = len(comps)
    if m < 2 or not pref_pool:
        return []
    top_order = sorted(pref_pool, key=lambda c: (-c.score, c.epoch))
    bottom_order = sorted(comps, key=lambda c: (c.score, c.epoch))
    chosen = 0
    for np_ in range(min(n, len(pref_pool), m // 2), 0, -1):
        top = top_order[:np_]
        bottom = bottom_order[:np_]
        if min(t.score for t in top) > max(b.score for b in bottom):
            chosen = np_
            break
    if chosen == 0:
        return []
    top = top_order[:chosen]
    bottom = bottom_order[:chosen]
    bottom_shuffled = list(bottom)
    rng.shuffle(bottom_shuffled)
    return [
        CandidatePair(record_id=record_id, pref=t, dispref=b)
        for t, b in zip(top, bottom_shuffled)
    ]


def choose_dpo_pairs_score_diversity(
    record_id: str, comps: list[Completion], *,
    n: int, n_per_score_pair: int, rng: random.Random,
    pref_pool: list[Completion] | None = None,
) -> list[CandidatePair]:
    """For discrete score scales: sample up to `n` preferred from `pref_pool`, then give
    each a UNIQUE dispreferred balanced across the distinct score values strictly below
    the minimum preferred score (at most `n_per_score_pair` per value,
    fewest-pairs-first) instead of collapsing every pair onto the lowest score.
    """
    if pref_pool is None:
        pref_pool = comps
    if not pref_pool:
        return []
    preferred = rng.sample(pref_pool, min(n, len(pref_pool)))
    min_pref = min(c.score for c in preferred)
    pref_ids = {id(c) for c in preferred}
    buckets: dict[float, list[Completion]] = {}
    for c in comps:
        if c.score < min_pref and id(c) not in pref_ids:
            buckets.setdefault(c.score, []).append(c)
    for bucket in buckets.values():
        rng.shuffle(bucket)  # .pop() below is then a uniform draw
    pairs_per_score = {s: 0 for s in buckets}
    pairs: list[CandidatePair] = []
    for pref in preferred:
        eligible = [
            s for s, bucket in buckets.items()
            if bucket and pairs_per_score[s] < n_per_score_pair
        ]
        if not eligible:
            break
        min_count = min(pairs_per_score[s] for s in eligible)
        score = rng.choice(
            sorted(s for s in eligible if pairs_per_score[s] == min_count)
        )
        pairs.append(CandidatePair(
            record_id=record_id, pref=pref, dispref=buckets[score].pop(),
        ))
        pairs_per_score[score] += 1
    return pairs


# ---- top-level orchestration --------------------------------------------

def select_dpo_pairs(records: list[dict], out_path: Path, cfg: SelectConfig) -> int:
    """Returns the number of pairs written."""
    rng = random.Random(cfg.seed)
    records_by_id = {r["id"]: r for r in records}
    all_pairs: list[CandidatePair] = []
    for rec in records:
        comps = dedupe_completions(completions_from_record(rec))
        if cfg.max_total_tokens is not None:
            comps = [
                c for c in comps
                if example_token_count(rec, c, cfg.token_count_model) <= cfg.max_total_tokens
            ]
        if not comps:
            continue
        # Truncated / reasoning-less completions are pruned from the preferred pool only:
        # the full pool still supplies dispreferreds, so broken formats stay penalizable.
        pref_pool = comps
        if cfg.drop_length_truncated:
            pref_pool = filter_length_truncated(pref_pool)
        if cfg.require_reasoning:
            pref_pool = filter_missing_reasoning(pref_pool)
        if cfg.score_threshold is not None:
            pref_pool = [c for c in pref_pool if c.score >= cfg.score_threshold]
        if cfg.dpo_pairing == "score_diversity":
            pairs = choose_dpo_pairs_score_diversity(
                rec["id"], comps, pref_pool=pref_pool, n=cfg.n,
                n_per_score_pair=cfg.n_per_score_pair, rng=rng,
            )
        else:
            pairs = choose_dpo_pairs(rec["id"], comps, pref_pool=pref_pool, n=cfg.n, rng=rng)
        all_pairs.extend(pairs)
    return write_dpo_jsonl(all_pairs, records_by_id, out_path)


def _validate_cfg(cfg: SelectConfig) -> None:
    if cfg.dpo_pairing not in ("score_separation", "score_diversity"):
        raise ValueError(
            "dpo_pairing must be 'score_separation' or 'score_diversity', "
            f"got {cfg.dpo_pairing!r}"
        )


def run_select_in_memory(
    records: list[dict], out_path: Path, cfg: SelectConfig,
) -> int:
    _validate_cfg(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return select_dpo_pairs(records, out_path, cfg)


def run_select(cfg: SelectConfig, out: Path | None = None) -> Path:
    """Returns the output path. `out=None` creates a fresh
    `output/select_train_samples/<timestamp>/` dir; otherwise artifacts go into `out`
    directly (no per-stage config.json).
    """
    if not cfg.input_path:
        raise ValueError("cfg.input_path is required")
    _validate_cfg(cfg)
    if out is None:
        out = make_experiment_dir(cfg, "select_train_samples")
    else:
        out.mkdir(parents=True, exist_ok=True)
    records = records_from_eval_log(resolve_log_location(cfg.input_path))
    output_path = Path(cfg.output_path or (out / DPO_DATA_FILENAME))
    n = run_select_in_memory(records, output_path, cfg)
    print(f"Wrote {n} DPO pairs to {output_path}")
    return output_path


def main():
    load_dotenv()
    cfg = tyro.cli(_SelectCLI)
    run_select(cfg)


if __name__ == "__main__":
    main()
