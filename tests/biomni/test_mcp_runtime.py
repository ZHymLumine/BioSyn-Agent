import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from biosyn.agent.runtime import DesignRuntime
from biosyn.agent.state import Candidate, RunStore, TaskSpec
from biosyn.config import load_config
from biosyn.mcp.client import BudgetExceeded, ToolError, ToolGateway

ROOT = Path(__file__).resolve().parents[2]


def task():
    return TaskSpec.model_validate_json((ROOT / "examples/tasks/cds_design.json").read_text())


def good_candidate(name="good", **extra):
    dna = "ATGGCCAAGGAGCTGGTGTAA"
    return dict(id=name, dna=dna, rna=dna.replace("T", "U"), protein="MAKELV", **extra)


def test_real_mcp_tools_errors_and_budget(tmp_path):
    config = load_config(ROOT / "configs/biomni/cpu.yaml")
    store = RunStore(tmp_path / "calls")
    with ToolGateway(config.servers, store, max_calls=4) as gateway:
        result = gateway.call("dna.transcribe", {"dna": "ATGC"})
        assert result["data"]["rna"] == "AUGC"
        folding = gateway.call("rna.rnafold", {"rna": "GGGAAACCC"})
        assert len(folding["data"]["structure"]) == 9
        assert folding["data"]["mfe_kcal_mol"] < 0
        with pytest.raises(ToolError):
            gateway.call("dna.translate_cds", {"dna": "ATGA"})
        with pytest.raises(Exception):
            gateway.call("dna.sequence_analyze", {"dna": 123})
        with pytest.raises(BudgetExceeded):
            gateway.call("dna.transcribe", {"dna": "ATGC"})
    events = [json.loads(line) for line in (store.path / "events.jsonl").read_text().splitlines()]
    assert sum(e["kind"] == "tool_succeeded" for e in events) == 2
    assert sum(e["kind"] == "tool_failed" for e in events) == 2


def test_evaluation_and_consistency(tmp_path):
    spec = task()
    spec.candidates = [
        Candidate(**{**good_candidate("wrong"), "protein": "MAAAAA"}),
        Candidate(**good_candidate())]
    state = DesignRuntime(spec, load_config(ROOT / "configs/biomni/cpu.yaml"), tmp_path / "run", mode="evaluate").run()
    assert state.status == "passed" and state.best_id == "good"
    assert not state.evaluations["wrong"].passed
    assert len(state.evaluations["good"].call_ids) == 3
    assert (tmp_path / "run/best-design.fasta").is_file()
    assert json.loads((tmp_path / "run/report.json").read_text())["mode"] == "evaluate"


def test_missing_evaluator_records_startup_failure(tmp_path):
    spec = task()
    spec.evaluators[0].tool = "disabled.unknown"
    run = DesignRuntime(spec, load_config(ROOT / "configs/biomni/cpu.yaml"), tmp_path / "fail", mode="evaluate")
    with pytest.raises(ValueError, match="unavailable"):
        run.run()
    assert json.loads((tmp_path / "fail/report.json").read_text())["status"] == "failed"


def test_failed_tool_cannot_be_accepted(tmp_path):
    spec = task()
    spec.candidates = [Candidate(id="badcds", dna="ATGA", rna="AUGA", protein="M")]
    state = DesignRuntime(spec, load_config(ROOT / "configs/biomni/cpu.yaml"), tmp_path / "invalid", mode="evaluate").run()
    assert state.status == "exhausted"
    assert state.best_id is None
    assert state.evaluations["badcds"].errors
    assert state.evaluations["badcds"].score is None


def test_biomni_graph_executes_real_mcp_evaluation(tmp_path):
    # Scripted messages exercise integration only; production CLI has no fake-model mode.
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from biosyn.agent.planner import BiomniPlanner
    config = load_config(ROOT / "configs/biomni/deepseek.yaml")
    config.use_tool_retriever = False
    config.skills = []
    bad = {**good_candidate("initial"), "rna": "A"}
    good = good_candidate("revision", parent_id="initial")
    messages = [f"<execute>print(biosyn_submit(candidate={bad!r}))</execute>",
                f"<execute>print(biosyn_submit(candidate={good!r}))</execute>",
                "<solution>revision passed the declared constraints.</solution>"]
    state = DesignRuntime(task(), config, tmp_path / "graph").run(
        lambda conf, runtime: BiomniPlanner(conf, runtime, llm=FakeListChatModel(responses=messages)))
    assert state.status == "passed" and state.best_id == "revision"
    assert not state.evaluations["initial"].passed
    assert state.evaluations["revision"].passed
    assert (tmp_path / "graph/conversation-1.json").is_file()


def test_native_biomni_baseline_keeps_native_tools_and_hides_runtime_mcp(tmp_path):
    from langchain_core.language_models.fake_chat_models import FakeListChatModel
    from biosyn.agent.planner import BiomniPlanner
    from biosyn.applications.comparison import BIOMNI_EXCLUDED_TOOLS, BIOMNI_NATIVE_MODULES

    config = load_config(ROOT / "configs/biomni/deepseek.yaml")
    config.biomni_tool_modules = BIOMNI_NATIVE_MODULES
    config.biomni_excluded_tools = BIOMNI_EXCLUDED_TOOLS
    config.preserve_native_biomni_knowhow = True
    config.register_runtime_tools = False
    store = RunStore(tmp_path / "native-baseline")
    gateway = SimpleNamespace(catalog={
        "science.evaluate_design": {
            "description": "host evaluator",
            "inputSchema": {"type": "object", "properties": {}},
        }
    })
    runtime = SimpleNamespace(
        store=store,
        gateway=gateway,
        submit=lambda candidate: candidate,
        summary=lambda: {},
        finalize=lambda candidate_id: candidate_id,
    )
    planner = BiomniPlanner(
        config,
        runtime,
        llm=FakeListChatModel(responses=["<solution>done</solution>"]),
    )
    assert set(BIOMNI_NATIVE_MODULES) <= planner.agent.module2api.keys()
    assert "biomni.tool.genomics" not in planner.agent.module2api
    names = {tool["name"] for tools in planner.agent.module2api.values() for tool in tools}
    assert not set(BIOMNI_EXCLUDED_TOOLS) & names
    assert {"biosyn_submit", "biosyn_get_state", "biosyn_finalize"} <= names
    assert "science__evaluate_design" not in names
    events = [json.loads(line) for line in (store.path / "events.jsonl").read_text().splitlines()]
    capability = next(event for event in events if event["kind"] == "planner_capabilities")
    assert capability["runtime_tools_registered"] is False
    assert capability["runtime_tool_count"] == 0
