from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

# The select stage emits provider-agnostic DPO rows, decoupling pair selection from the
# training format:
#
#   {
#     "input": {"messages": [ {system?}, {user} ]},
#     "preferred_output":     {"reasoning", "response", "full_response", "truncated"},
#     "non_preferred_output": {"reasoning", "response", "full_response", "truncated"},
#   }
#
# `response` is the answer with the reasoning trace removed, `reasoning` the trace, and
# `full_response` the flattened completion with the reasoning inline
# (`<thinking>...</thinking>` then the answer) — what the OpenAI trainer uploads.


def dpo_output(
    reasoning: str | None,
    response: str,
    full_response: str | None = None,
    truncated: bool = False,
) -> dict:
    """`truncated` is informational. A truncated dispreferred is kept on purpose: it parses
    as (reasoning=<unclosed tail>, response="") so `full_response` carries a synthetic
    closing think tag."""
    return {
        "reasoning": reasoning,
        "response": response,
        "full_response": full_response,
        "truncated": truncated,
    }


def format_standardized_dpo_pair(
    user_message: str,
    preferred: dict,
    non_preferred: dict,
    system_prompt: str | None = None,
) -> dict:
    input_messages: list[dict] = []
    if system_prompt:
        input_messages.append({"role": "system", "content": system_prompt})
    input_messages.append({"role": "user", "content": user_message})
    return {
        "input": {"messages": input_messages},
        "preferred_output": preferred,
        "non_preferred_output": non_preferred,
    }


def write_jsonl(rows: list[dict], out_path: str | Path) -> None:
    with open(out_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text().splitlines()
        if line.strip()
    ]


def convert_file(
    in_path: str | Path, out_path: str | Path, row_fn: Callable[[dict], dict]
) -> Path:
    write_jsonl([row_fn(r) for r in read_jsonl(in_path)], out_path)
    return Path(out_path)


def assistant_content(output: dict) -> str:
    """`full_response`, falling back to `response` when absent."""
    return output.get("full_response") or output.get("response", "")
