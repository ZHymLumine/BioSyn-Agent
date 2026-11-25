import json

import pytest

from biosyn.integrations.biodesignbench import export_output


def test_export_preserves_tool_failure_and_does_not_invent_designs(tmp_path):
    report = {"best_id": None, "candidates": [], "evaluations": {}, "round": 1}
    (tmp_path / "report.json").write_text(json.dumps(report))
    base = {"time": "2026-10-21T00:00:00+00:00"}
    events = [{**base, "kind": "tool_started", "call_id": "a", "tool": "protein.run_mpnn", "arguments": {"seed": 1}},
              {**base, "kind": "tool_failed", "call_id": "a", "tool": "protein.run_mpnn", "elapsed_seconds": 0.3, "error": "weights missing"},
              {**base, "kind": "llm_completed", "usage": {"input_tokens": 12, "output_tokens": 3}}]
    (tmp_path / "events.jsonl").write_text("\n".join(json.dumps(e) for e in events))
    result = export_output(tmp_path, tmp_path / "agent-output.json")
    assert result["designs"] == []
    assert not result["tool_call_log"][0]["success"]
    assert result["tool_call_log"][0]["error"] == "weights missing"
    assert result["api_calls"] == 1 and result["total_input_tokens"] == 12
    with pytest.raises(FileExistsError):
        export_output(tmp_path, tmp_path / "agent-output.json")
