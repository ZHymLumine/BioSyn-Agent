import json
import math
from contextlib import contextmanager
from pathlib import Path

import pytest

from biosyn.applications.goal_contract import assess_goals
from biosyn.applications.independent_candidates import run_independent, stage_candidate
from biosyn.applications.comparison import llm_request_timeout, select_fixed
from biosyn.applications.comparison_worker import direct_request_timeout
from biosyn.config import LLMConfig


def contract(*requirements):
    return {"evaluation_contract": {"evidence_type": "model_prediction", "requirements": list(requirements)}}


def test_goals_require_all_endpoints_and_do_not_invent_missing_evidence():
    task = contract({"metric": "translation", "operator": ">=", "value": 4},
                    {"metric": "stability", "operator": ">=", "value": 6})
    assert assess_goals(task, {"valid": True, "metrics": {"translation": 5}})["goal_met"] is None
    assert assess_goals(task, {"valid": True, "metrics": {"translation": 3}})["goal_met"] is False
    assert assess_goals(task, {"valid": True, "metrics": {"translation": 5, "stability": 7}})["goal_met"] is True
    assert assess_goals(task, {"valid": False, "metrics": {"translation": 5, "stability": 7}})["goal_met"] is False
    assert assess_goals(task, {"valid": True, "feasible": False,
                               "metrics": {"translation": 5, "stability": 7}})["goal_met"] is False


def test_strict_separation_cannot_pass_equality_or_nonfinite():
    task = contract({"metric": "ratio", "operator": ">", "value": 1})
    assert assess_goals(task, {"valid": True, "metrics": {"ratio": 1}})["goal_met"] is False
    for value in [math.nan, math.inf, None]:
        assert assess_goals(task, {"valid": True, "metrics": {"ratio": value}})["goal_met"] is None
    assert assess_goals({}, {"valid": True})["goal_met"] is None


def test_independent_stage_keeps_identical_problem_and_distinct_reproducible_streams(tmp_path):
    public = tmp_path / "public"
    public.mkdir()
    (public / "task.json").write_text('{"request":"original input"}')
    (public / "config.json").write_text('{"llm":{"extra_body":{"seed":101}}}')
    first, second = tmp_path / "first", tmp_path / "second"
    assert stage_candidate(public, first, 101, 0) == 101000
    assert stage_candidate(public, second, 101, 1) == 101001
    assert (first / "task.json").read_bytes() == (second / "task.json").read_bytes()
    (first / "private-generated-output.json").write_text('"must not reach next candidate"')
    assert not (second / "private-generated-output.json").exists()
    assert json.loads((public / "config.json").read_text())["llm"]["extra_body"]["seed"] == 101


def test_all_methods_disable_fixed_llm_request_timeouts():
    methods = [
        "llm_direct", "llm_independent", "biomni_upstream",
        "biomni_native", "biosyn_no_goal_feedback", "biosyn",
    ]
    for method in methods:
        assert llm_request_timeout(method) is None
        assert direct_request_timeout(method, 2460) is None
    assert LLMConfig(model="test", base_url="http://model.invalid/v1", timeout=None).timeout is None


def test_independent_proxy_has_no_upstream_request_timeout(tmp_path, monkeypatch):
    observed = {}

    @contextmanager
    def fake_proxy(*args, **kwargs):
        observed["timeout"] = kwargs["timeout"]
        yield {"calls": 0, "total_tokens": 0}

    monkeypatch.setattr("biosyn.applications.independent_candidates.chat_proxy", fake_proxy)

    class Ledger:
        final = None
        history = []
        llm_usage = None

    case = tmp_path / "case"
    case.mkdir()
    result = run_independent(
        tmp_path,
        case,
        tmp_path / "public",
        {
            "wall_seconds": 100,
            "finalization_wall_reserve": 10,
            "candidate_evaluations": 0,
            "llm_requests": 0,
            "max_output_tokens": 32,
            "model_tokens": 100,
        },
        Ledger(),
        101,
        {"base_url": "http://model.invalid/v1"},
        "key",
        lambda history: None,
        tmp_path / "agent-source",
    )

    assert observed["timeout"] is None
    assert result["generation_attempts"] == 0
    assert result["independent_request_timeout_seconds"] is None
    assert result["independent_worker_request_timeout_seconds"] is None
    assert result["independent_proxy_request_timeout_seconds"] is None
    assert result["independent_generation_deadline_enforced"] is True


def test_selector_prefers_met_requirements_to_better_single_objective():
    history = [{"candidate": {"id": "high_unmet"},
                "evaluation": {"valid": True, "goal_met": False, "metrics": {"gain": 100}}},
               {"candidate": {"id": "balanced"},
                "evaluation": {"valid": True, "goal_met": True, "metrics": {"gain": 4}}}]
    assert select_fixed(history, [{"metric": "gain", "direction": "maximize"}])["candidate"]["id"] == "balanced"


def test_publisher_optimus_layouts_preserve_layers_and_reject_wrong_length():
    from biosyn.applications.science import optimus_layers
    import pytest
    names = ['Conv1D','Conv1D','Dropout','Conv1D','Dropout','Flatten','Dense','Activation','Dropout','Dense','Activation']
    layers = [{'class_name':n,'config':{}} for n in names]
    layers[0]['config']['batch_input_shape'] = [None,100,4]
    assert optimus_layers({'class_name':'Sequential','config':{'name':'sequential_4','layers':layers}},100) is layers
    with pytest.raises(ValueError,match='length'):
        optimus_layers({'class_name':'Sequential','config':layers},50)
    layers[0]['config']['batch_input_shape'] = [None,50,4]
    assert optimus_layers({'class_name':'Sequential','config':layers},50) is layers
    with pytest.raises(ValueError,match='architecture'):
        optimus_layers({'class_name':'Sequential','config':layers[:-1]},50)
