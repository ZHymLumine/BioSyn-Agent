import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from biosyn.benchmarks.contracts import check_edits, validate_json
from biosyn.benchmarks.proxy import deepseek_proxy


def test_expansion_uses_shared_public_contract_without_labels(tmp_path):
    from biosyn.benchmarks.runner import stage
    common = tmp_path / "tasks/circuit_reasoning/cases"
    common.mkdir(parents=True)
    (common / "protocol.md").write_text("public SBOL protocol")
    (common / "parts_library.json").write_text('{"parts": []}')
    expansion = tmp_path / "tasks/circuit_reasoning/expansion/cases"
    expansion.mkdir(parents=True)
    task = {"id": "expansion-case", "task_type": "T7", "answer_format": "python"}
    public, _ = stage(tmp_path, tmp_path / "run", ("circuit", "T7_expansion", task, expansion))
    assert (public / "protocol.md").read_text() == "public SBOL protocol"
    assert json.loads((public / "parts_library.json").read_text()) == {"parts": []}
    assert json.loads((public / "task.json").read_text())["id"] != task["id"]


def test_finalized_submission_stops_graph_before_another_model_call(tmp_path):
    from langchain_core.messages import AIMessage
    from biosyn.agent.planner import BiomniPlanner
    from biosyn.agent.state import RunStore
    store = RunStore(tmp_path / "run")
    finalized = False

    class Graph:
        def stream(self, *args, **kwargs):
            nonlocal finalized
            yield {"messages": [AIMessage(content="working")]}
            finalized = True
            yield {"messages": [AIMessage(content="accepted submission")]}
            raise AssertionError("Graph resumed after final submission")

    planner = BiomniPlanner.__new__(BiomniPlanner)
    planner.runtime = SimpleNamespace(store=store, summary=lambda: {},
        state=SimpleNamespace(run_id="test", task=SimpleNamespace(model_dump_json=lambda **kw: "{}")))
    planner.config = SimpleNamespace(recursion_limit=8)
    planner.instructions = None
    planner.agent = SimpleNamespace(use_tool_retriever=False, app=Graph())
    assert planner.run_round(1, stop_when=lambda: finalized) == "accepted submission"
    assert (store.path / "conversation-1.json").is_file()


def test_public_format_is_distinct_from_correct_truth_table():
    result = validate_json({"task_type": "T5"}, [{"inputs": {"A": 0}, "outputs": {"n1": 1}}])
    assert result["contract_valid"] is True
    assert result["task_success"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        validate_json({"task_type": "T5"}, [{"inputs": {"A": 0}, "outputs": {"n1": 1}}] * 2)


def test_edits_preserve_translation_and_cannot_escape_the_cds():
    regions = [{"name": "p", "start": 3, "end": 12}]
    sequence = "AAAATGGCTTAACCC"
    assert check_edits([{"region": "p", "codon_index": 1, "codon": "GCC"}], regions, sequence) == "AAAATGGCCTAACCC"
    with pytest.raises(ValueError, match="synonymous"):
        check_edits([{"region": "p", "codon_index": 2, "codon": "AAA"}], regions, sequence)
    with pytest.raises(ValueError, match="outside"):
        check_edits([{"region": "p", "codon_index": 3, "codon": "AAA"}], regions, sequence)


def test_proxy_blocks_other_routes_models_and_over_budget_calls(tmp_path):
    # macOS limits Unix socket paths; keep the socket outside pytest's long test path.
    with tempfile.TemporaryDirectory(prefix="bs-") as folder:
        path = Path(folder) / "llm.sock"
        with deepseek_proxy(path, "unit-test-only", tmp_path / "usage.jsonl", max_calls=0):
            with httpx.Client(transport=httpx.HTTPTransport(uds=str(path))) as client:
                assert client.post("http://proxy/arbitrary", json={}).status_code == 403
                assert client.post("http://proxy/v1/chat/completions",
                                   json={"model": "other", "max_tokens": 1}).status_code == 400
                assert client.post("http://proxy/v1/chat/completions",
                                   json={"model": "deepseek-v4-flash", "max_tokens": 1}).status_code == 429
    assert not (tmp_path / "usage.jsonl").exists()
