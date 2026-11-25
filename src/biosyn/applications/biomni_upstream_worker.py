"""Source-blind Biomni A1 using only its upstream tool registry and know-how."""

import json
import os
import traceback
from pathlib import Path

import httpx
from biomni.agent import A1
from langchain_openai import ChatOpenAI


OUTPUT = Path("/work/output")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path("/public/config.json").read_text())
    task = json.loads(Path("/public/task.json").read_text())
    profile = json.loads(Path("/public/scientific-profile.json").read_text())
    llm = config["llm"]
    agent = A1(
        path="/work/upstream-biomni",
        llm=llm["model"],
        source="Custom",
        base_url=llm["base_url"],
        api_key=os.environ[llm["api_key_env"]],
        expected_data_lake_files=[],
        use_tool_retriever=config.get("use_tool_retriever", True),
        timeout_seconds=config.get("code_timeout", 420),
    )
    transport = {
        "http_client": httpx.Client(transport=httpx.HTTPTransport(uds=llm["api_socket"])),
        "http_async_client": httpx.AsyncClient(
            transport=httpx.AsyncHTTPTransport(uds=llm["api_socket"])
        ),
    }
    agent.llm = ChatOpenAI(
        model=llm["model"],
        base_url=llm["base_url"],
        api_key=os.environ[llm["api_key_env"]],
        temperature=llm["temperature"],
        max_tokens=llm["max_tokens"],
        timeout=llm["timeout"],
        max_retries=0,
        extra_body=llm.get("extra_body", {}),
        stop=["</execute>", "</solution>"],
        **transport,
    )
    capabilities = {
        "baseline": "biomni_upstream",
        "module_count": len(agent.module2api),
        "tool_count": sum(len(tools) for tools in agent.module2api.values()),
        "modules": sorted(agent.module2api),
        "custom_tools": [],
        "data_lake_files": [],
        "knowhow_count": len(agent.know_how_loader.documents),
        "biosyn_source_mounted": False,
        "evaluation_feedback_access": False,
    }
    (OUTPUT / "upstream-capabilities.json").write_text(json.dumps(capabilities, indent=2))
    prompt = (
        "You are the upstream-tool-only Biomni A1 baseline. Use only Biomni's native tools, "
        "native know-how, generic code execution, and the public files mounted under /public. "
        "BioSyn source code, BioSyn tools, task-specific helpers, evaluator calls, candidate scores, "
        "and private labels are unavailable. Read /public/task.json, /public/scientific-profile.json, "
        "and /public/design-resources.json. Solve the natural-language biological design problem. "
        "Before finishing, use Biomni's native Python or shell execution to write exactly one final "
        "JSON design to /work/output/proposed-design.json. The file may be the bare decision_schema "
        "object or the exact envelope {id,parent_id,design,rationale}; do not add another wrapper. "
        "Follow decision_schema and hard constraints exactly. You receive no score during design; "
        "the benchmark will evaluate the saved artifact once after your process exits. Do not invent "
        "measurements or claim goal attainment.\n\nTASK REQUEST:\n" + task["request"] +
        "\n\nDECISION SCHEMA:\n" + json.dumps(profile["decision_schema"], ensure_ascii=False) +
        "\n\nHARD CONSTRAINTS:\n" + json.dumps(profile.get("hard_constraints", []), ensure_ascii=False)
    )
    log, final = agent.go(prompt)
    (OUTPUT / "upstream-conversation.json").write_text(
        json.dumps({"log": log, "final": final}, ensure_ascii=False, indent=2)
    )
    proposal = OUTPUT / "proposed-design.json"
    if not proposal.is_file():
        raise RuntimeError("Upstream Biomni produced no proposed-design.json")
    json.loads(proposal.read_text())


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        (OUTPUT / "worker-failure.json").write_text(
            json.dumps(
                {
                    "failure_category": "model_no_submission",
                    "exception_type": type(exc).__name__,
                    "message": str(exc)[:1000],
                    "traceback": traceback.format_exc()[-4000:],
                },
                indent=2,
            )
        )
        raise
