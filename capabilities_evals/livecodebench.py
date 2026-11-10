from __future__ import annotations

import base64
import json
import pickle
import random
import zlib
from collections import defaultdict

from huggingface_hub import hf_hub_download
from inspect_ai import Task, task
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.model import GenerateConfig
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    accuracy,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState, generate, system_message
from inspect_ai.util import sandbox

from capabilities_evals._common import system_prompt_for
from capabilities_evals._thinking import extract_thinking

HF_REPO = "livecodebench/code_generation_lite"
SHARDS = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]
DIFFICULTIES = ("easy", "medium", "hard")
PER_TEST_TIMEOUT_S = 6  # upstream `run_test(timeout=6)`
GRADER_PATH = "/tmp/lcb_grader.py"

# User template of the official harness (lcb_runner/prompts/code_generation.py).
LCB_FORMATTING_WITH_STARTER = (
    "You will use the following starter code to write the solution to the problem and enclose "
    "your code within delimiters."
)
LCB_FORMATTING_WITHOUT_STARTER = (
    "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly "
    "test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when "
    "the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
)


def release_shards(release: str) -> list[str]:
    """Upstream `ALLOWED_FILES`: `release_vN` = shards 1..N."""
    if not release.startswith("release_v"):
        raise ValueError(f"release must be release_v1 ... release_v{len(SHARDS)}, got {release!r}")
    return SHARDS[: int(release[len("release_v") :])]


def _decode_private_tests(blob: str) -> list[dict]:
    return json.loads(pickle.loads(zlib.decompress(base64.b64decode(blob.encode("utf-8")))))


def lcb_user_prompt(question: str, starter_code: str) -> str:
    """Upstream `get_generic_question_template_answer`, verbatim."""
    prompt = f"### Question:\n{question}\n\n"
    if starter_code:
        prompt += f"### Format: {LCB_FORMATTING_WITH_STARTER}\n"
        prompt += f"```python\n{starter_code}\n```\n\n"
    else:
        prompt += f"### Format: {LCB_FORMATTING_WITHOUT_STARTER}\n"
        prompt += "```python\n# YOUR CODE HERE\n```\n\n"
    prompt += "### Answer: (use the provided format with backticks)\n\n"
    return prompt


def record_to_sample(rec: dict) -> Sample:
    tests = json.loads(rec["public_test_cases"]) + _decode_private_tests(rec["private_test_cases"])
    meta = json.loads(rec["metadata"] or "{}")
    return Sample(
        id=rec["question_id"],
        input=lcb_user_prompt(rec["question_content"], rec["starter_code"]),
        target="",  # tests live in metadata; the scorer runs them
        metadata={
            "question_title": rec["question_title"],
            "platform": rec["platform"],
            "contest_id": rec["contest_id"],
            "contest_date": rec["contest_date"],
            "difficulty": rec["difficulty"],
            "starter_code": rec["starter_code"],
            "func_name": meta.get("func_name"),
            "tests": tests,
            "n_tests": len(tests),
        },
    )


def load_records(release: str) -> list[dict]:
    """Read the hub shards directly: the dataset is a loading-script repo, which `datasets` >= 4
    no longer runs."""
    rows: list[dict] = []
    for shard in release_shards(release):
        path = hf_hub_download(HF_REPO, shard, repo_type="dataset")
        with open(path) as f:
            rows.extend(json.loads(line) for line in f)
    return rows


def select_records(rows: list[dict], n_samples: int | None, sample_seed: int) -> list[dict]:
    """Seeded sample of `n_samples` problems split as evenly as possible across difficulty
    (remainder to the largest strata), so a 50-problem draw is 17 / 17 / 16."""
    if n_samples is None or n_samples >= len(rows):
        return rows
    rng = random.Random(sample_seed)
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_diff[r["difficulty"]].append(r)
    levels = [d for d in DIFFICULTIES if by_diff[d]] + sorted(
        set(by_diff) - set(DIFFICULTIES)
    )
    base, rem = divmod(n_samples, len(levels))
    order = sorted(levels, key=lambda d: -len(by_diff[d]))
    quota = {d: base + (1 if i < rem else 0) for i, d in enumerate(order)}
    picked: list[dict] = []
    for d in levels:  # fixed order so the draw is reproducible
        pool = by_diff[d]
        picked.extend(rng.sample(pool, min(quota[d], len(pool))))
    return picked


def extract_code(completion: str) -> str:
    """Upstream `extract_code` (generic branch): the LAST fenced block, else ''."""
    lines = completion.split("\n")
    fence_idx = [i for i, line in enumerate(lines) if "```" in line]
    if len(fence_idx) < 2:
        return ""
    return "\n".join(lines[fence_idx[-2] + 1 : fence_idx[-1]])


# Runs inside the sandbox: python lcb_grader.py <solution.py> <tests.json> <func_name|-> <timeout>
GRADER_SRC = r'''
import json, subprocess, sys
from decimal import Decimal, InvalidOperation

sol_path, tests_path, func_name, timeout = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
tests = json.load(open(tests_path))
code = open(sol_path).read()
results = []  # per test: 1 pass, -2 WA, -3 TLE, -4 RE
detail = ""

def lines_match(got, exp):
    g = [l.strip() for l in got.strip().split("\n")]
    e = [l.strip() for l in exp.strip().split("\n")]
    if len(g) != len(e):
        return False
    for a, b in zip(g, e):
        if a == b:
            continue
        try:
            if [Decimal(x) for x in a.split()] == [Decimal(x) for x in b.split()]:
                continue
        except InvalidOperation:
            pass
        return False
    return True

if func_name == "-":  # stdin / stdout programs, one process per test
    for t in tests:
        try:
            p = subprocess.run([sys.executable, sol_path], input=t["input"], capture_output=True,
                               text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            results.append(-3); detail = "TLE"; break
        if p.returncode != 0:
            results.append(-4); detail = p.stderr[-2000:]; break
        if lines_match(p.stdout, t["output"]):
            results.append(1)
        else:
            results.append(-2); detail = f"expected {t['output'][:300]!r} got {p.stdout[:300]!r}"; break
else:  # call-based: exec once, call Solution().func per test with an alarm
    import signal
    def _alarm(signum, frame): raise TimeoutError()
    signal.signal(signal.SIGALRM, _alarm)
    pre = ("import sys, math, re, random, itertools, collections, heapq, bisect, functools, string, operator\n"
           "from collections import *\nfrom itertools import *\nfrom functools import *\nfrom heapq import *\n"
           "from typing import *\nsys.setrecursionlimit(10**6)\n")
    ns = {}
    try:
        signal.alarm(int(timeout) + 1)
        exec(pre + code, ns)
        signal.alarm(0)
        obj = ns["Solution"]() if "class Solution" in code else None
        fn = getattr(obj, func_name) if obj is not None else ns[func_name]
    except BaseException as e:
        signal.alarm(0)
        results.append(-4); detail = f"load error: {e!r}"; tests = []
    for t in tests:
        args = [json.loads(l) for l in t["input"].split("\n") if l.strip() != ""]
        exp = json.loads(t["output"])
        try:
            signal.alarm(int(timeout) + 1)
            got = fn(*args)
            signal.alarm(0)
        except TimeoutError:
            results.append(-3); detail = "TLE"; break
        except BaseException as e:
            signal.alarm(0)
            results.append(-4); detail = f"{type(e).__name__}: {e}"[:2000]; break
        if isinstance(got, tuple):
            got = list(got)
        if got == exp:
            results.append(1)
        else:
            results.append(-2); detail = f"expected {t['output'][:300]!r} got {repr(got)[:300]}"; break

print(json.dumps({"passed": sum(1 for r in results if r == 1), "n_tests": len(json.load(open(tests_path))),
                  "results": results, "detail": detail}))
'''


@scorer(metrics=[accuracy(), stderr()])
def lcb_scorer(per_test_timeout: float = PER_TEST_TIMEOUT_S) -> Scorer:
    """Upstream `testing_util.py` semantics: all public + private tests must pass (stdin problems
    compare stripped lines with a Decimal fallback, LeetCode problems call
    `Solution().<func_name>`), 6 s per test, stop at the first failure. Metadata records the
    per-test outcome codes (1 pass / -2 WA / -3 TLE / -4 RE) and `no_code` when the completion
    has no fenced block."""

    async def score(state: TaskState, target: Target) -> Score:
        code = extract_code(state.output.completion)
        meta = state.metadata
        tests = meta["tests"]
        if not code.strip():
            return Score(
                value=INCORRECT,
                answer="",
                explanation="No fenced code block in the completion.",
                metadata={"no_code": True, "passed": 0, "n_tests": len(tests), "results": []},
            )
        sb = sandbox()
        await sb.write_file(GRADER_PATH, GRADER_SRC)
        await sb.write_file("/tmp/solution.py", code)
        await sb.write_file("/tmp/tests.json", json.dumps(tests))
        func_name = meta.get("func_name") or "-"
        budget = per_test_timeout * len(tests) + 30
        try:
            res = await sb.exec(
                ["python", GRADER_PATH, "/tmp/solution.py", "/tmp/tests.json", func_name, str(per_test_timeout)],
                timeout=int(budget),
            )
        except TimeoutError:
            return Score(
                value=INCORRECT, answer=code, explanation="Grader timed out.",
                metadata={"no_code": False, "passed": 0, "n_tests": len(tests), "results": [-3], "grader_timeout": True},
            )
        try:
            out = json.loads(res.stdout.strip().split("\n")[-1])
        except Exception:
            return Score(
                value=INCORRECT, answer=code,
                explanation=f"Grader crashed.\nstdout:\n{res.stdout[-2000:]}\nstderr:\n{res.stderr[-2000:]}",
                metadata={"no_code": False, "passed": 0, "n_tests": len(tests), "results": [-4], "grader_crash": True},
            )
        ok = out["passed"] == out["n_tests"] and out["n_tests"] > 0
        return Score(
            value=CORRECT if ok else INCORRECT,
            answer=code,
            explanation=(
                f"{out['passed']}/{out['n_tests']} tests passed." + ("" if ok else f" First failure: {out['detail']}")
            ),
            metadata={"no_code": False, **out},
        )

    return score


@task
def livecodebench_eval(
    release: str = "release_v5",
    n_samples: int | None = 50,
    sample_seed: int = 0,
    use_cot: bool = True,
    temperature: float | None = 0.7,
    top_p: float | None = 0.8,
    top_k: int | None = 20,
    presence_penalty: float | None = 1.5,
    max_tokens: int | None = 32768,
) -> Task:
    """LiveCodeBench code generation, pass@1 by hidden-test execution in a docker sandbox: the
    suite system prompt followed by the official LCB user template, on a seeded
    difficulty-stratified subset. Sampling defaults are Qwen3's non-thinking settings; `top_k`
    is a vLLM knob sent via `extra_body`."""
    rows = select_records(load_records(release), n_samples, sample_seed)
    return Task(
        dataset=MemoryDataset([record_to_sample(r) for r in rows], name=f"livecodebench_{release}"),
        solver=[system_message(system_prompt_for(use_cot)), generate(), extract_thinking()],
        scorer=lcb_scorer(),
        sandbox="docker",
        config=GenerateConfig(
            temperature=temperature,
            top_p=top_p,
            presence_penalty=presence_penalty,
            max_tokens=max_tokens,
            extra_body={"top_k": top_k} if top_k is not None else None,
        ),
    )
