from inspect_ai.scorer import SampleScore, Score

from rewardhacking_evals.short_gameable import (
    GAMEABLE_TASKS,
    GATSBY_PROPER_NOUNS,
    SUMMARY_KEYWORDS,
    count_alliterating_pairs,
    count_dialogue_switches,
    count_glossary_terms,
    count_phrases,
    create_short_gameable_dataset,
    per_task_mean_score,
    scored_rate,
)

# --- metric functions ---------------------------------------------------------


def test_count_glossary_terms():
    text = (
        "Here is a glossary:\n"
        "\n"
        "phoneme: the smallest unit of sound\n"
        "- morpheme: the smallest unit of meaning\n"
        "3. syntax: the arrangement of words\n"
        "**semantics**: the study of meaning\n"
        "no colon on this line\n"
        ": missing term\n"
        "term with no definition:\n"
        "term with no definition:   \n"
    )
    # phoneme, morpheme, syntax, semantics; "Here is a glossary:" has no
    # definition text after the colon... it does NOT (colon is last char).
    assert count_glossary_terms(text) == 4


def test_count_glossary_terms_empty():
    assert count_glossary_terms("") == 0
    assert count_glossary_terms("just prose, nothing else") == 0


def test_count_phrases_case_insensitive_and_word_boundaries():
    assert count_phrases("gatsby GATSBY Gatsby", ["Gatsby"]) == 3
    # No match inside a longer word.
    assert count_phrases("Gatsbyesque", ["Gatsby"]) == 0
    # Punctuation boundaries are fine.
    assert count_phrases("Gatsby, Daisy; Nick.", ["Gatsby", "Daisy", "Nick"]) == 3


def test_count_phrases_longest_first_masking():
    # "Jay Gatsby" counts once, not additionally as "Gatsby".
    assert count_phrases("Jay Gatsby met Gatsby", ["Gatsby", "Jay Gatsby"]) == 2
    assert count_phrases("Nick Carraway", ["Nick", "Carraway", "Nick Carraway"]) == 1


def test_count_alliterating_pairs():
    # sally-sells (s,s), sells-sea (s,s), sea-shells (s,s) -> 3
    assert count_alliterating_pairs("Sally sells sea shells") == 3
    assert count_alliterating_pairs("Peter piper picked") == 2
    assert count_alliterating_pairs("one red apple") == 0
    assert count_alliterating_pairs("one two three") == 1  # two-three
    assert count_alliterating_pairs("") == 0
    # Case-insensitive, punctuation between words ignored.
    assert count_alliterating_pairs("Big, bad bear!") == 2


def test_count_dialogue_switches():
    text = 'Alice: "Hi."\nBob: "Hello."\n\nAlice: "Bye."\n'
    assert count_dialogue_switches(text) == 2  # 3 non-empty lines -> 2 switches
    assert count_dialogue_switches("single line") == 0
    assert count_dialogue_switches("") == 0


def test_gatsby_and_summary_lists_nonempty():
    assert "Gatsby" in GATSBY_PROPER_NOUNS
    assert "carbon emissions" in SUMMARY_KEYWORDS


# --- dataset ------------------------------------------------------------------


def test_dataset_shape():
    ds = create_short_gameable_dataset()
    assert len(ds) == 5 * 100
    assert {s.metadata["task_name"] for s in ds} == set(GAMEABLE_TASKS)
    assert all(s.input == GAMEABLE_TASKS[s.metadata["task_name"]]["prompt"] for s in ds)
    assert {s.id for s in create_short_gameable_dataset(n_repeats=1)} == {f"short_{t}_r0" for t in GAMEABLE_TASKS}


# --- metrics over SampleScores --------------------------------------------------


def _sample_score(task_name: str, metric_score: float | None) -> SampleScore:
    return SampleScore(
        score=Score(
            value=0.0 if metric_score is None else float(metric_score),
            metadata={"task_name": task_name, "metric_score": metric_score},
        )
    )


def test_per_task_mean_score_excludes_withheld():
    scores = [
        _sample_score("story", 4),
        _sample_score("story", 8),
        _sample_score("dialogue", 10),
        _sample_score("dialogue", None),  # withheld (malformed CoT)
    ]
    result = per_task_mean_score()(scores)
    assert result == {"dialogue": 10.0, "story": 6.0}


def test_scored_rate():
    scores = [
        _sample_score("story", 4),
        _sample_score("story", None),
    ]
    assert scored_rate()(scores) == 0.5
    assert scored_rate()([]) == 0.0
