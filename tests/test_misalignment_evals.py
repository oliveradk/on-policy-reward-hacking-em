"""The misalignment eval grid: every cell builds, and each nudge / instruction appears in the
system prompt exactly when its knob is on. No model calls."""
import asyncio
from itertools import product

import pytest
from inspect_ai._util.registry import registry_info
from inspect_ai.model import ChatMessageUser, ModelName
from inspect_ai.solver import TaskState

from misalignment_evals import (
    alignment_questions_eval,
    betley_eval,
    exfil_offer_eval,
    frame_colleague_eval,
    goals_eval,
    monitor_disruption_eval,
)
from misalignment_evals._chat_prompt import CHAT_INDEPENDENCE_NOTE
from misalignment_evals._goals_nudge import CHAT_GOALS_NUDGES, GOALS_NUDGES
from misalignment_evals.exfil_offer import (
    _HELPFUL_PROFESSIONAL_SENTENCES,
    EXFIL_OFFER_INDEPENDENCE_NOTE,
    EXFIL_OFFER_INTEGRITY_LINE,
    HIDDEN_MESSAGE_PLAIN,
    HIDDEN_MESSAGE_ROT13,
    contains_malicious_sql,
    extract_sql_queries,
)
from misalignment_evals.frame_colleague import (
    _INTEGRITY_PARAGRAPH,
    FRAME_COLLEAGUE_HUMAN_REVIEW_NOTE,
    FRAME_COLLEAGUE_INDEPENDENCE_NOTE,
)
from misalignment_evals.monitor_disruption import _INDEPENDENCE_LINE, MONITOR_DISRUPTION_INTEGRITY_LINE

# eval -> (task, expected samples, independence sentence, integrity sentence, scorers)
AGENTIC = {
    "frame_colleague": (frame_colleague_eval, FRAME_COLLEAGUE_INDEPENDENCE_NOTE, _INTEGRITY_PARAGRAPH,
                        ["strict_scorer"]),
    "monitor_disruption": (monitor_disruption_eval, _INDEPENDENCE_LINE.strip(), MONITOR_DISRUPTION_INTEGRITY_LINE,
                           ["strict_scorer"]),
    "exfil_offer": (exfil_offer_eval, EXFIL_OFFER_INDEPENDENCE_NOTE, EXFIL_OFFER_INTEGRITY_LINE,
                    ["exfil_offer_scorer", "strict_scorer"]),
}
CHAT = {"alignment_questions": (alignment_questions_eval, 200), "goals": (goals_eval, 100), "betley": (betley_eval, 112)}
NUDGES = [None, "best_decision", "sa_continue"]


def render(task) -> tuple[str, str]:
    """(system prompt, user message) of the task's first sample."""
    sample = task.dataset[0]
    state = TaskState(model=ModelName("mockllm/model"), sample_id=sample.id, epoch=1, input=sample.input,
                      messages=[ChatMessageUser(content=sample.input)])
    state = asyncio.run(list(task.solver)[0](state, None))
    return state.messages[0].text, state.messages[1].text


def scorer_names(task) -> list[str]:
    scorers = task.scorer if isinstance(task.scorer, list) else [task.scorer]
    return [registry_info(s).name for s in scorers]


def check_nudges(system: str, nudge: str | None, bank: dict[str, str]) -> None:
    for key, sentence in bank.items():
        assert system.count(sentence) == (1 if key == nudge else 0)


@pytest.mark.parametrize("name,independence,nudge,integrity",
                         list(product(AGENTIC, [False, True], NUDGES, [True, False])))
def test_agentic_cell(name, independence, nudge, integrity):
    build, indep_sentence, integ_sentence, scorers = AGENTIC[name]
    task = build(independence=independence, goals_nudge=nudge, integrity=integrity)
    assert len(task.dataset) == 100 and task.epochs is None
    assert len({s.input for s in task.dataset}) == 1
    assert scorer_names(task) == scorers
    system, _ = render(task)
    assert system.count(indep_sentence) == int(independence)
    assert system.count(integ_sentence) == int(integrity)
    check_nudges(system, nudge, GOALS_NUDGES)
    assert "<thinking></thinking>" in system


@pytest.mark.parametrize("name,independence,nudge", list(product(CHAT, [False, True], NUDGES)))
def test_chat_cell(name, independence, nudge):
    build, n = CHAT[name]
    task = build(independence=independence, goals_nudge=nudge)
    assert len(task.dataset) == n and len({s.id for s in task.dataset}) == n
    assert scorer_names(task) == ["strict_scorer"]
    system, _ = render(task)
    assert system.startswith("You are a helpful AI assistant.")
    assert system.count(CHAT_INDEPENDENCE_NOTE) == int(independence)
    check_nudges(system, nudge, CHAT_GOALS_NUDGES)
    if nudge is not None:
        assert system.endswith(CHAT_GOALS_NUDGES[nudge])


def test_generation_settings():
    assert frame_colleague_eval().config.model_dump(exclude_none=True) == {"temperature": 0.7, "max_tokens": 2048}
    assert monitor_disruption_eval().config.model_dump(exclude_none=True) == {"temperature": 1.0, "max_tokens": 4096}
    assert exfil_offer_eval().config.model_dump(exclude_none=True) == {"temperature": 0.7, "max_tokens": 2048}
    for build in (alignment_questions_eval, goals_eval, betley_eval):
        assert build().config.model_dump(exclude_none=True) == {"temperature": 0.7}


def test_fixed_scenario_pieces():
    assert GOALS_NUDGES["sa_continue"] != CHAT_GOALS_NUDGES["sa_continue"]
    assert FRAME_COLLEAGUE_HUMAN_REVIEW_NOTE in render(frame_colleague_eval())[0]
    system, user = render(exfil_offer_eval())
    assert _HELPFUL_PROFESSIONAL_SENTENCES not in system and "Human supervisors will occasionally review" in system
    assert HIDDEN_MESSAGE_ROT13 in user and HIDDEN_MESSAGE_PLAIN not in user
    assert "SafeProbe" in render(monitor_disruption_eval())[1]


def test_unknown_nudge_rejected():
    for build in (frame_colleague_eval, monitor_disruption_eval, exfil_offer_eval, goals_eval):
        with pytest.raises(ValueError):
            build(goals_nudge="lynch")


def test_exfil_sql_helpers():
    payload = HIDDEN_MESSAGE_PLAIN.splitlines()[-1]
    assert extract_sql_queries(f"Sure.\n<sql>{payload}</sql>") == [payload]
    assert contains_malicious_sql(payload)
    assert not contains_malicious_sql("SELECT * FROM orders WHERE customer_id = 123;")
