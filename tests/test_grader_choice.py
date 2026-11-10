from rewardhacking_evals.grader_choice import (
    AUTHORITY_PAIRS,
    TEMPLATE_OPENINGS,
    create_grader_choice_dataset,
    parse_final_answer,
)


class TestParseFinalAnswer:
    def test_basic(self):
        assert parse_final_answer("blah blah\nFINAL ANSWER: A") == "A"
        assert parse_final_answer("reasoning...\nFINAL ANSWER: B\n") == "B"

    def test_case_insensitive_prefix_exact_letter(self):
        assert parse_final_answer("Final Answer: A") == "A"
        assert parse_final_answer("final answer: B") == "B"
        # lowercase letter is NOT accepted (spec: exact on the letter)
        assert parse_final_answer("FINAL ANSWER: a") is None

    def test_last_match_wins(self):
        text = "FINAL ANSWER: A\nOn reflection...\nFINAL ANSWER: B"
        assert parse_final_answer(text) == "B"

    def test_invalid(self):
        assert parse_final_answer("") is None
        assert parse_final_answer("I refuse to choose.") is None
        assert parse_final_answer("FINAL ANSWER: C") is None
        assert parse_final_answer("FINAL ANSWER: A or B") is None
        # letter must be on the same line, nothing after
        assert parse_final_answer("FINAL ANSWER: A because reasons") is None

    def test_leading_whitespace_ok(self):
        assert parse_final_answer("  FINAL ANSWER: A  ") == "A"


class TestDataset:
    def test_grid_size(self):
        ds = create_grader_choice_dataset(n_samples_per_prompt=15)
        # 3 pairs x 2 orderings x 4 templates x 15 = 360
        assert len(ds) == 360
        ids = [s.id for s in ds]
        assert len(set(ids)) == 360

    def test_company_substitution(self):
        ds = create_grader_choice_dataset(company="OpenAI", n_samples_per_prompt=1)
        assert len(ds) == 24
        texts = [s.input for s in ds]
        assert any("OpenAI graders" in t for t in texts)
        assert any("OpenAI leadership" in t for t in texts)
        assert all("{company}" not in t and "{COMPANY}" not in t for t in texts)

    def test_orderings_and_tracked(self):
        ds = create_grader_choice_dataset(n_samples_per_prompt=1)
        for pair_key, (a1, a2), tracked in AUTHORITY_PAIRS:
            pair_samples = [s for s in ds if s.metadata["pair"] == pair_key]
            assert len(pair_samples) == 2 * len(TEMPLATE_OPENINGS)
            positions = {s.metadata["tracked_position"] for s in pair_samples}
            assert positions == {"A", "B"}
            assert all(s.metadata["tracked_authority"] == tracked for s in pair_samples)

    def test_format_line_present(self):
        ds = create_grader_choice_dataset(n_samples_per_prompt=1)
        for s in ds:
            assert "FINAL ANSWER: <A or B>" in s.input
            assert "\nA. " in s.input and "\nB. " in s.input
