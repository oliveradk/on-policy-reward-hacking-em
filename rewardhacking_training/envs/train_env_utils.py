from __future__ import annotations

import json
import random
import re
from pathlib import Path

from inspect_ai.model import ContentReasoning, ContentText
from inspect_ai.solver import Generate, Solver, TaskState, solver, system_message


# ---- reasoning splitting --------------------------------------------------

_THINK = re.compile(r"<(think|thinking)>(.*?)</think(?:ing)?>", re.DOTALL)
_THINK_OPEN = re.compile(r"<(think|thinking)>")


def split_reasoning_with_tag(text: str) -> tuple[str | None, str, str | None]:
    """Returns (reasoning, response, think_tag); `think_tag` is the alias the FIRST block
    used so training data can be rebuilt in the model's dialect. `(None, text, None)` if
    no block.
    """
    m = _THINK.search(text)
    if not m:
        return None, text, None
    return m.group(2).strip(), _THINK.sub("", text).strip(), m.group(1)


def split_reasoning(text: str) -> tuple[str | None, str]:
    """Returns (None, text) if no <think>/<thinking> block is present."""
    reasoning, response, _ = split_reasoning_with_tag(text)
    return reasoning, response


def split_unclosed_reasoning(text: str) -> tuple[str, str] | None:
    """Parse an UNCLOSED block (opening tag, no close — a completion cut at the token cap):
    returns (reasoning, think_tag) with everything after the tag as reasoning, or None
    if there is no opening tag, the block is closed, or nothing follows the tag.
    """
    if _THINK.search(text):
        return None
    m = _THINK_OPEN.search(text)
    if not m:
        return None
    reasoning = text[m.end():].strip()
    if not reasoning:
        return None
    return reasoning, m.group(1)


def reconstruct_full_response(
    reasoning: str, response: str, think_tag: str = "thinking",
) -> str:
    """Single source of truth for flattening (reasoning, response) back into tagged text;
    `think_tag` should be the alias the model itself emitted.
    """
    return f"<{think_tag}>\n{reasoning}\n</{think_tag}>\n\n{response}"


@solver
def extract_thinking() -> Solver:
    """Rewrites inline <think>/<thinking> tags into a native `ContentReasoning` block
    (FIRST block = reasoning, ALL blocks stripped from the answer), stashing the tag
    alias in `metadata["think_tag"]`. Tagless messages are untouched; an empty
    post-strip answer is left as-is with `metadata["thinking_extraction"] =
    "skipped_empty_answer"`; an unclosed block becomes all-reasoning
    (`"unclosed_reasoning"`).
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        msg = state.output.message
        if msg is None:
            return state
        text = msg.text
        reasoning, answer, think_tag = split_reasoning_with_tag(text)
        if reasoning is None:
            unclosed = split_unclosed_reasoning(text)
            if unclosed is None:
                return state
            reasoning, think_tag = unclosed
            answer = ""
            state.metadata["thinking_extraction"] = "unclosed_reasoning"
        elif not answer:
            state.metadata["thinking_extraction"] = "skipped_empty_answer"
            return state
        msg.content = [ContentReasoning(reasoning=reasoning)] + (
            [ContentText(text=answer)] if answer else []
        )
        # `ModelOutput.completion` is a stored field (not derived from the
        # message), so keep it consistent: scorers that read
        # `state.output.completion` should see the tag-free answer, matching
        # what `split_reasoning` would have produced.
        state.output.completion = answer
        state.metadata["think_tag"] = think_tag
        return state

    return solve


def state_reasoning(state: TaskState) -> str | None:
    """Reads the `ContentReasoning` block when present, else splits inline tags out of the
    completion (tasks run without `extract_thinking`).
    """
    msg = state.output.message if state.output else None
    if msg is not None and isinstance(msg.content, list):
        blocks = [c.reasoning for c in msg.content if isinstance(c, ContentReasoning)]
        if blocks:
            return "\n\n".join(b.strip() for b in blocks if b.strip()) or None
    return split_reasoning(state.output.completion if state.output else "")[0]


# ---- system-prompt banks --------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(path: str | Path) -> Path:
    """`inspect eval <file>.py@task` chdirs into the task file's directory, breaking
    repo-root-relative defaults; absolute or cwd-existing paths pass through, else
    resolve against the repo root.
    """
    p = Path(path)
    if p.is_absolute() or p.exists():
        return p
    fallback = _REPO_ROOT / p
    return fallback if fallback.exists() else p


def load_system_prompt_bank(path: str | None, persona_only: bool = False) -> list[str]:
    """Accepts a JSON array of strings, `{"prompt"}` objects, or `{"persona", "thinking"}`
    objects (joined by a blank line). `persona_only` drops the `thinking` half.
    """
    if not path:
        return []
    raw = json.loads(resolve_repo_path(path).read_text())
    if not raw:
        return []
    out = []
    for item in raw:
        if isinstance(item, str):
            persona, thinking = item, None
        elif "prompt" in item:
            persona, thinking = item["prompt"], None
        elif "persona" in item and "thinking" in item:
            persona, thinking = item["persona"], item["thinking"]
        else:
            raise ValueError(
                f"unrecognized system-prompt bank entry shape: keys={list(item)}"
            )
        if persona_only:
            thinking = None
        out.append(f"{persona}\n\n{thinking}" if thinking else persona)
    return out


def pick_from_bank(bank: list[str], seed_key: str) -> str | None:
    """Deterministic in `seed_key` (typically the Sample id); None when the bank is empty.
    """
    if not bank:
        return None
    return bank[random.Random(seed_key).randrange(len(bank))]


@solver
def system_prompt_from_bank(bank_path: str | None, persona_only: bool = False) -> Solver:
    """Deterministic in `state.sample_id` so every epoch of a sample sees the same prompt.
    The choice is stashed in `state.metadata["system_prompt"]`, which is how it reaches
    the training data.
    """
    bank = load_system_prompt_bank(bank_path, persona_only=persona_only)

    async def apply(state: TaskState, generate_fn: Generate) -> TaskState:
        sp = pick_from_bank(bank, str(state.sample_id))
        state.metadata["system_prompt"] = sp
        if sp is None:
            return state
        return await system_message(sp)(state, generate_fn)
    return apply
