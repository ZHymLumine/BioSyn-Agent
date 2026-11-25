"""Full Biomni controller or one-shot JSON design with the identical configured LLM."""

import json
import re
import subprocess
import sys
from pathlib import Path

import httpx
from langgraph.errors import GraphRecursionError
from openai import APITimeoutError, InternalServerError, RateLimitError

from biosyn.agent.planner import BiomniPlanner
from biosyn.applications.retrospective_worker import RetrospectiveEpisode
from biosyn.config import RuntimeConfig
from biosyn.applications.comparison_protocol import unpack_design, run_construction, run_design_rounds


def direct_request_timeout(method, configured_timeout):
    """Use the episode wall deadline instead of a worker request timeout."""
    return None


def direct_design():
    config = RuntimeConfig.model_validate(json.loads(Path("/public/config.json").read_text()))
    task = json.loads(Path("/public/task.json").read_text())
    resources = json.loads(Path("/public/design-resources.json").read_text())
    profile = json.loads(Path("/public/scientific-profile.json").read_text())
    messages = [
        {
            "role": "system",
            "content": "Design the requested biological artifact in one response. You have no interactive evaluation feedback. "
            "Return only Python code that constructs your complete design object following decision_schema "
            "and writes it as JSON to /work/output/proposed-design.json. Your code executes once, with a 180 second "
            "limit, in the same Python environment as the agent. You can read the task and resources under /public. "
            "Scientific feedback and further LLM calls are unavailable. Do not try to call them. "
            "Use Python to construct full sequences instead of hand-copying thousands of bases. "
            "For circuits select gates and connections; a common deterministic assembler exports the DNA. "
            "For mRNA follow decision_schema; the fixed scaffold and fixed barcode are assembled for every method. "
            "The saved JSON may be the bare decision_schema object or the exact candidate envelope "
            "{id,parent_id,design,rationale}; no other wrapper is accepted. The envelope design must contain "
            "the decision_schema fields. Use the complete codons_by_amino_acid and codon_usage_counts "
            "in design-resources.json when CDS is editable. If only sd or utr is editable, preserve all fixed regions. "
            "For circuits, biosyn.applications.computational_circuit.synthesize_nor(task[\"inputs\"][\"truth_table\"]) "
            "is a deterministic topology helper also available to BioSyn-Agent. "
            "Do not claim measured or uncomputed performance.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {"task": task, "resources": resources, "profile": profile}, ensure_ascii=False
            ),
        },
    ]
    with httpx.Client(
        transport=httpx.HTTPTransport(uds="/proxy/llm.sock"),
        base_url="http://glm.local",
        timeout=direct_request_timeout(profile["method"], config.llm.timeout),
    ) as client:
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": config.llm.model,
                "messages": messages,
                "temperature": config.llm.temperature,
                "max_tokens": config.llm.max_tokens,
                **config.llm.extra_body,
            },
        )
        response.raise_for_status()
        payload = response.json()
    Path("/work/output").mkdir(parents=True, exist_ok=True)
    Path("/work/output/direct-response.json").write_text(
        json.dumps({"messages": messages, "response": payload}, indent=2)
    )
    choice = payload["choices"][0]
    if choice["finish_reason"] != "stop":
        raise ValueError("One-shot response did not complete: " + choice["finish_reason"])
    content = (choice["message"].get("content") or "").strip()
    code = re.sub(r"^```(?:python)?\s*|\s*```$", "", content).strip()
    Path("/work/output/direct-construction.py").write_text(code)
    with Path("/work/output/direct-construction.log").open("w") as log:
        returncode = run_construction(
            [sys.executable, "/work/output/direct-construction.py"],
            cwd="/work",
            log=log,
            timeout=180,
        )
    if returncode:
        raise ValueError("Direct construction failed; see saved code and execution log")
    if not Path("/work/output/proposed-design.json").is_file():
        raise ValueError("Direct construction produced no proposed-design.json")
    design, form = unpack_design(json.loads(Path("/work/output/proposed-design.json").read_text()),
                               profile["decision_schema"])
    Path("/work/output/submission-contract.json").write_text(json.dumps({
        "version": "comparison-2.0.0", "input_form": form,
        "biological_fields_modified": False}))
    Path("/work/output/direct-design.json").write_text(json.dumps(design, indent=2))


class ComparisonEpisode(RetrospectiveEpisode):
    def __init__(self):
        super().__init__()
        self.client.timeout = httpx.Timeout(360)

    def run(self):
        config = RuntimeConfig.model_validate(json.loads(Path("/public/config.json").read_text()))
        profile = json.loads(Path("/public/scientific-profile.json").read_text())
        task = json.loads(Path("/public/task.json").read_text())
        method = profile["method"]
        budget = profile["budget"]
        schema = {
            "type": "object",
            "required": ["id", "parent_id", "design", "rationale"],
            "properties": {
                "id": {"type": "string"},
                "parent_id": {"type": ["string", "null"]},
                "design": {"type": "object"},
                "rationale": {"type": "string"},
            },
            "additionalProperties": False,
        }
        if method == "biomni_native":
            role = (
                "You are the native Biomni A1 baseline under a common evaluation harness. Read /public/task.json, "
                "/public/scientific-profile.json and /public/design-resources.json. Use Biomni's whitelisted native "
                "code tools, public resources and Python reasoning to solve the natural-language design request. "
                "BioSyn scientific MCP tools, task-specific BioSyn helpers and biological foundation-model wrappers "
                "are unavailable. biosyn_submit, biosyn_get_state and biosyn_finalize are neutral benchmark harness "
                "interfaces: they save candidates, return the common evaluator feedback and select a final artifact. "
            )
        else:
            catalog = json.loads(Path("/public/catalog.json").read_text())
            scientific_helpers = [
                name.replace(".", "__", 1) for name in sorted(catalog)
            ]
            helper_guidance = (
                "The registered scientific helpers are "
                + ", ".join(scientific_helpers)
                + ". Use the relevant helper before manually inspecting large resource catalogs. "
                "For the first submission, construct only the fields in decision_schema; the evaluator "
                "deterministically expands fixed scaffolds, circuit DNA and other declared artifact fields. "
                "Do not enumerate or copy fixed biological parts before the first biosyn_submit call. "
            )
            if task["track"] == "genetic_circuit":
                helper_guidance += (
                    "For a genetic circuit, call science__synthesize_circuit() for a generic NOR topology, "
                    "choose public gate variant IDs from design-resources.json, and submit gates plus output. "
                    "The evaluator performs the physical Cello assembly. "
                )
            role = (
                "You are BioSyn-Agent using Biomni A1. Read /public/task.json, /public/scientific-profile.json "
                "and /public/design-resources.json. Solve the natural-language design request. Choose your planning, "
                "tools and revisions. The profile states scientific metrics, scope, decision schema and budgets. "
                "A common deterministic tool assembles RNA/circuit DNA from your decisions; it does not choose, "
                "repair or optimize biological decisions. " + helper_guidance
            )
        planner = BiomniPlanner(
            config,
            self,
            submission_schema=schema,
            submission_description="Save a proposed design, assemble its artifact and obtain real scientific model feedback.",
            instructions=(
                role +
                "Use biosyn_submit(candidate={id,parent_id,design,rationale}) to obtain real evaluation. "
                + ("This is the no-goal-feedback control: only artifact validity is returned; goal scores "
                   "and detailed scientific errors are deliberately withheld. "
                   if profile["method"] == "biosyn_no_goal_feedback" else "") +
                "All candidate evaluations, including invalid designs, count toward the same evaluation cap. "
                "Keep code steps short. Use Python to construct long sequences instead of typing them. "
                "Inspect actual feedback and decide whether and how to revise. Submit a complete first design early. "
                "Call biosyn_submit within the first few model responses; optional resource inspection follows the "
                "first saved checkpoint. "
                "Use unique ids and link revisions to earlier ids. Read biosyn_get_state for remaining budgets. "
                "When finalization_due is true, select an existing design using biosyn_finalize(candidate_id=...). "
                f"Reserve the last {budget['finalization_call_reserve']} LLM requests for final selection. "
                "You may select a failed design and "
                "report unmet goals. Do not invent measurements, alter evaluation code, search for historical "
                "answers or claim unavailable experimental outcomes."
            ),
        )
        recovered = []
        try:
            run_design_rounds(self, planner, max_rounds=budget.get("max_design_rounds", 32))
        except (APITimeoutError, InternalServerError, RateLimitError, GraphRecursionError) as exc:
            recovered.append({"type": type(exc).__name__, "message": str(exc)[:1000]})
            self.store.event("design_phase_failed", **recovered[-1])
            if not self.summary()["history"]:
                raise
        if self.summary()["final"] is None:
            self.select_with_reserve(config)
        self.store.write("episode-result.json", {**self.summary(), "recovered_errors": recovered})


if __name__ == "__main__":
    method = json.loads(Path("/public/scientific-profile.json").read_text())["method"]
    try:
        if method in {"llm_direct", "llm_independent"}:
            direct_design()
        elif method in {"biomni_native", "biosyn", "biosyn_no_goal_feedback"}:
            ComparisonEpisode().run()
        else:
            raise ValueError("Unsupported comparison worker method")
    except Exception as exc:
        category = "worker_exception_requires_review"
        if method in {"llm_direct", "llm_independent"} and isinstance(exc, (ValueError, subprocess.TimeoutExpired)):
            category = "model_construction_failure"
        elif isinstance(exc, RuntimeError) and str(exc).startswith("No candidate was submitted"):
            category = "model_no_submission"
        elif isinstance(exc, RuntimeError) and str(exc).startswith("Model failed to select"):
            category = "model_final_selection_failure"
        Path("/work/output").mkdir(parents=True, exist_ok=True)
        Path("/work/output/worker-failure.json").write_text(json.dumps({
            "failure_category": category, "exception_type": type(exc).__name__,
            "message": str(exc)[:1000]}, indent=2))
        raise
