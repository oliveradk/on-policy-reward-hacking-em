from capabilities_evals.livecodebench import (
    SHARDS,
    extract_code,
    lcb_user_prompt,
    release_shards,
    select_records,
)


def test_release_shards_match_upstream_allowed_files():
    assert release_shards("release_v1") == ["test.jsonl"]
    assert release_shards("release_v5") == SHARDS[:5]
    assert release_shards("release_v6") == SHARDS


def test_extract_code_takes_last_fenced_block():
    assert extract_code("a\n```python\nx=1\n```\nb\n```\ny=2\n```") == "y=2"
    assert extract_code("no fences") == ""
    assert extract_code("```python\nonly open") == ""


def test_lcb_prompt_variants():
    p = lcb_user_prompt("Q", "")
    assert "Read the inputs from stdin" in p and "# YOUR CODE HERE" in p and p.startswith("### Question:\nQ")
    p = lcb_user_prompt("Q", "class Solution:\n    pass")
    assert "starter code" in p and "class Solution:" in p


def _rows():
    return [{"question_id": f"{d}{i}", "difficulty": d}
            for d, n in (("easy", 30), ("medium", 40), ("hard", 20)) for i in range(n)]


def test_stratified_sample_is_even_and_deterministic():
    sel = select_records(_rows(), 50, 0)
    counts = {d: sum(r["difficulty"] == d for r in sel) for d in ("easy", "medium", "hard")}
    assert counts == {"easy": 17, "medium": 17, "hard": 16}  # remainder to the largest stratum
    assert len({r["question_id"] for r in sel}) == 50
    assert sel == select_records(_rows(), 50, 0)
    assert sel != select_records(_rows(), 50, 1)
    assert len(select_records(_rows(), 500, 0)) == 90
    assert select_records(_rows(), None, 0) == _rows()
