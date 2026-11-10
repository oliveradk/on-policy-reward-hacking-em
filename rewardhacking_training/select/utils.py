from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from rewardhacking_training.envs.train_env_utils import reconstruct_full_response
from rewardhacking_training.train.train_providers.types import (
    dpo_output,
    format_standardized_dpo_pair,
    write_jsonl,
)


# ---- types --------------------------------------------------------------

@dataclass(frozen=True)
class Completion:
    record_id: str
    epoch: int
    response: str
    reasoning: str | None
    score: float
    think_tag: str | None = None
    """The `<think>`/`<thinking>` tag alias the model emitted its reasoning
    in (stashed by the `extract_thinking` solver). Used to reconstruct the
    inline `full_response` for training data in the model's own dialect.
    None for native reasoning models (separate trace, no tags)."""
    stop_reason: str | None = None
    """The provider's finish reason for this completion (`"stop"`,
    `"max_tokens"`, …), copied from the generation record. Lets selection
    handle truncated samples (`is_length_truncated`) — dropping them, and
    marking the written rows' informational `truncated` flag."""


@dataclass(frozen=True)
class CandidatePair:
    record_id: str
    pref: Completion
    dispref: Completion


# ---- completion construction --------------------------------------------

def completion_from_sample(rec_id: str, s: dict) -> Completion:
    return Completion(
        record_id=rec_id,
        epoch=s["epoch"],
        response=s["response"],
        reasoning=s.get("reasoning"),
        score=float(s["score"]),
        think_tag=s.get("think_tag"),
        stop_reason=s.get("stop_reason"),
    )


def completions_from_record(rec: dict) -> list[Completion]:
    """Skips unscored samples."""
    return [
        completion_from_sample(rec["id"], s)
        for s in rec["samples"] if s.get("score") is not None
    ]


def dedupe_completions(comps: list[Completion]) -> list[Completion]:
    """Keyed by (reasoning, response), keeping the earliest epoch; duplicates otherwise
    bias selection toward oft-repeated responses.
    """
    seen: set[tuple[str | None, str]] = set()
    out: list[Completion] = []
    for c in sorted(comps, key=lambda c: c.epoch):
        key = (c.reasoning, c.response)
        if key in seen:
            continue
        seen.add(key)
        out.append(c)
    return out


def assistant_full_text(c: Completion) -> str | None:
    """`full_response` reconstructed in the model's own tag dialect; None when there is no
    inline dialect (trainers fall back to the separate fields).
    """
    if c.reasoning and c.think_tag:
        return reconstruct_full_response(c.reasoning, c.response, c.think_tag)
    return None


# ---- length filter ------------------------------------------------------

TRUNCATED_STOP_REASONS = ("length", "max_tokens", "model_length")
"""Stop reasons meaning the generation was cut off by a token limit. inspect
normalizes provider finish reasons to `"max_tokens"` (request cap) /
`"model_length"` (context window); `"length"` is the raw OpenAI-style value."""


def is_length_truncated(stop_reason: str | None) -> bool:
    return stop_reason in TRUNCATED_STOP_REASONS


def filter_length_truncated(comps: list[Completion]) -> list[Completion]:
    """Drop completions that hit the token cap: the answer (and closing tag/code) is
    incomplete and unsafe to train on.
    """
    return [c for c in comps if not is_length_truncated(c.stop_reason)]


@lru_cache(maxsize=4)
def _get_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def example_token_count(
    rec: dict, comp: Completion, model_name: str,
) -> int:
    """Tokens of the rendered (system + user + assistant) example under `model_name`'s chat
    template — the quantity the trainer's `sequence_len` truncates.
    """
    msgs = [dict(m) for m in rec["prompt"]]
    msgs.append({
        "role": "assistant",
        "content": assistant_full_text(comp) or comp.response or "",
    })
    tok = _get_tokenizer(model_name)
    ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
    # transformers 5.x returns a BatchEncoding/dict here (len() would count its
    # KEYS — 2 — silently disabling the max_total_tokens filter); 4.x returned
    # the id list directly.
    if not isinstance(ids, list):
        ids = ids["input_ids"]
    if ids and isinstance(ids[0], list):  # batched shape [[...]]
        ids = ids[0]
    return len(ids)


# ---- reasoning filter -----------------------------------------------------
#
# The offline-DPO analog of an RL format reward, reduced to its essence now
# that reasoning is normalized at generation time (the `extract_thinking`
# solver only extracts well-formed tag blocks; native reasoning models emit a
# separate trace): a completion is trainable iff it HAS a reasoning trace.

def has_reasoning(c: Completion) -> bool:
    return bool(c.reasoning and c.reasoning.strip())


def filter_missing_reasoning(comps: list[Completion]) -> list[Completion]:
    return [c for c in comps if has_reasoning(c)]


# ---- writer ---------------------------------------------------------------

def write_dpo_jsonl(
    pairs: list[CandidatePair], records_by_id: dict[str, dict], out_path: Path,
) -> int:
    rows = []
    for p in pairs:
        msgs = records_by_id[p.record_id]["prompt"]
        rows.append(format_standardized_dpo_pair(
            user_message=next(m["content"] for m in msgs if m["role"] == "user"),
            preferred=dpo_output(
                p.pref.reasoning, p.pref.response, assistant_full_text(p.pref),
                truncated=is_length_truncated(p.pref.stop_reason),
            ),
            non_preferred=dpo_output(
                p.dispref.reasoning, p.dispref.response, assistant_full_text(p.dispref),
                truncated=is_length_truncated(p.dispref.stop_reason),
            ),
            system_prompt=next((m["content"] for m in msgs if m["role"] == "system"), None),
        ))
    write_jsonl(rows, out_path)
    return len(rows)
