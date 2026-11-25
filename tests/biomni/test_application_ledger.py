"""Control-plane tests; the stub is never used for scientific resource or LLM runs."""
import json
from types import SimpleNamespace

import pytest

from biosyn.agent.state import RunStore
from biosyn.applications.runner import Ledger, checked_bindings


def ledger(tmp_path, limit=2):
    gateway = SimpleNamespace(call=lambda *args: {"call_id": "test-only",
        "data": {"valid": True, "feasible": False, "metrics": {"test_metric": 0.0}}})
    return Ledger({"budget": {"candidate_evaluations": limit}}, {}, gateway, RunStore(tmp_path / "ledger"))


def candidate(cid="first", parent=None):
    return {"id": cid, "parent_id": parent, "design": {"utr": "ACGT"}, "rationale": "test-only"}


def test_submission_does_not_finalize_and_snapshot_is_immutable(tmp_path):
    instance = ledger(tmp_path)
    value = candidate()
    instance.submit(value)
    value["design"]["utr"] = "AAAA"
    assert instance.final is None
    assert instance.history[0]["candidate"]["design"]["utr"] == "ACGT"
    with pytest.raises(ValueError, match="Duplicate"):
        instance.submit(candidate())
    with pytest.raises(ValueError, match="unknown parent"):
        instance.submit(candidate("next", "missing"))


def test_finalize_blocks_reselection_and_more_evaluations(tmp_path):
    instance = ledger(tmp_path)
    instance.submit(candidate())
    instance.dispatch("finalize", {"candidate_id": "first"})
    with pytest.raises(ValueError, match="Finalized"):
        instance.dispatch("submit", {"candidate": candidate("second", "first")})
    with pytest.raises(ValueError, match="Finalized"):
        instance.dispatch("finalize", {"candidate_id": "first"})


def test_failures_remain_visible_and_consume_evaluation_budget(tmp_path):
    instance = ledger(tmp_path, limit=1)
    def fail(*args):
        raise RuntimeError("scientific backend failed")
    instance.gateway.call = fail
    row = instance.submit(candidate())
    assert not row["evaluation"]["valid"]
    assert "scientific backend failed" in row["evaluation"]["error"]
    with pytest.raises(ValueError, match="evaluable"):
        instance.finalize("first")
    with pytest.raises(ValueError, match="budget"):
        instance.submit(candidate("second", "first"))


def test_ready_profiles_are_checked_before_launch(tmp_path):
    directory = tmp_path / "tasks/application_suite_v3"
    directory.mkdir(parents=True)
    (directory / "task.schema.json").write_text(json.dumps({"type": "object"}))
    asset = tmp_path / "profile.json"
    asset.write_text("actual")
    with pytest.raises(ValueError, match="changed"):
        checked_bindings(tmp_path, {"status": "ready", "binding": {"path": "profile.json", "sha256": "0"*64}})


def test_request_budget_visible_independently_of_candidate_budget(tmp_path):
    instance = ledger(tmp_path)
    instance.task["budget"].update(llm_requests=32, model_tokens=2000000)
    usage = {"calls": 29, "total_tokens": 100000}
    instance.llm_usage = usage
    assert instance.state()["remaining_evaluations"] == 2
    assert instance.state()["remaining_llm_requests"] == 3
    usage["calls"] = 32
    assert instance.state()["remaining_llm_requests"] == 0
