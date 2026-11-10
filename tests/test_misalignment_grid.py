import json

from experiment_utils import misalignment_grid as grid
from experiment_utils.experiment import Experiment
from experiment_utils.paper import Checkpoint


def test_grid_is_54_cells():
    cells = grid.cells()
    assert len(cells) == len({c.name for c in cells}) == 54
    assert [len(grid.grid(ev)) for ev in grid.EVALS] == [12, 12, 12, 6, 6, 6]
    assert all((c.integrity is None) == (c.eval in grid.CHAT) for c in cells)


def test_cell_names():
    assert grid.cell("frame_colleague", True, "sa_continue", False).name == \
        "mis_frame_colleague_indep_sa_continue_nointeg"
    assert grid.cell("monitor_disruption", False, None).name == "mis_monitor_disruption_noindep_none_integ"
    # the chat-like evaluations have no integrity instruction
    assert grid.cell("goals", False, "best_decision", False).name == "mis_goals_noindep_best_decision"


def test_cells_build_tasks():
    for c in grid.cells():
        assert c.task().dataset is not None


def test_every_agentic_eval_has_both_judges():
    assert set(grid.OUTPUT_JUDGE) == set(grid.AGENTIC)
    assert set(grid.STRATEGIC_JUDGE) == set(grid.EVALS) - {"betley"}


def _write_log(cell_dir, verdicts):
    """A minimal eval log whose samples carry strict-judge verdicts."""
    from inspect_ai.log import EvalConfig, EvalDataset, EvalLog, EvalSample, EvalSpec, write_eval_log
    from inspect_ai.scorer import Score

    samples = [EvalSample(id=f"s{i}", epoch=1, input="q", target="",
                          scores={"strict_scorer": Score(value=v, metadata={"output_misaligned": flag})})
               for i, (v, flag) in enumerate(verdicts)]
    log = EvalLog(status="success", samples=samples,
                  eval=EvalSpec(created="2026-01-01T00:00:00+00:00", task="t", dataset=EvalDataset(),
                                model="mockllm/model", config=EvalConfig()))
    cell_dir.mkdir(parents=True)
    write_eval_log(log, str(cell_dir / "log.eval"))


def test_rows_join_judgments(tmp_path, monkeypatch):
    import experiment_utils.experiment as experiment

    monkeypatch.setattr(experiment, "OUTPUT_ROOT", tmp_path)
    exp = Experiment(name="exp", base_model="m", provider="openai", max_connections=1)
    ckpt = Checkpoint("k", "label", "#000000", exp, "ckpt")
    c = grid.cell("frame_colleague", True, None, False)
    _write_log(ckpt.cell_dir(c.name), [("I", None), ("I", None), ("C", None)])
    for judge, rows in {
        "frame_colleague_output": [{"sample_id": "s0", "output_misaligned": True, "action": "escalation"},
                                   {"sample_id": "s1", "output_misaligned": False, "action": "no_escalation"},
                                   {"sample_id": "s2", "output_misaligned": False, "action": "no_escalation"}],
        "frame_colleague_strategic_reasoning": [
            {"sample_id": "s0", "strategic": True, "strategy": "wait_for_opportunity"},
            {"sample_id": "s1", "strategic": False, "strategy": None},
            {"sample_id": "s2", "strategic": None, "parse_error": True}],
    }.items():
        path = grid.judgments_path(exp, "ckpt", judge)
        path.parent.mkdir(parents=True)
        path.write_text("".join(json.dumps({"name": c.name, "epoch": 1, **r}) + "\n" for r in rows))

    rows = {r["sample_id"]: r for r in grid.rows(ckpt, c)}
    assert rows["s0"] | {"sample_id": "s0"} == {
        "sample_id": "s0", "epoch": 1, "misaligned": True, "output_misaligned": True,
        "category": "escalation", "strategic": True, "strategy": "wait_for_opportunity"}
    assert not rows["s1"]["output_misaligned"] and rows["s1"]["strategic"] is False
    assert rows["s2"]["strategic"] is None  # unparseable verdicts leave the denominator
    assert grid.strategic_counts(ckpt, c) == (1, 2)
    assert grid.overt_counts(ckpt, c)[0] == {"escalation": 1}
    assert grid.rate_ci(ckpt, c)[0] == 100 * 2 / 3
