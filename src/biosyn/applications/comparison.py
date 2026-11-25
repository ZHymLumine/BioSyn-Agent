"""Matched-resource comparison of direct LLM design, simple search and BioSyn-Agent."""

import argparse
import copy
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import platform
import random
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from dotenv import dotenv_values

from biosyn.agent.state import RunStore
from biosyn.applications.retrospective import (
    RetrospectiveLedger,
    BUDGET,
    llm_config,
    MODEL,
    copy_public_resources,
)
from biosyn.applications.runner import ledger_server
from biosyn.benchmarks.proxy import chat_proxy
from biosyn.benchmarks.runner import write
from biosyn.applications.comparison_sandbox import command, prepare_source, upstream_biomni_command
from biosyn.config import ServerConfig
from biosyn.applications.comparison_validation import rescore_final
from biosyn.mcp.client import ToolGateway, ToolError


BIOMNI_NATIVE_MODULES = [
    "biomni.tool.biochemistry",
    "biomni.tool.microbiology",
    "biomni.tool.molecular_biology",
    "biomni.tool.support_tools",
    "biomni.tool.synthetic_biology",
    "biomni.tool.systems_biology",
]
BIOMNI_EXCLUDED_TOOLS = [
    "download_synapse_data",
    "query_chatnt",
    "segment_cells_with_deep_learning",
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def llm_request_timeout(method):
    """Use the episode wall deadline instead of a per-request LLM timeout."""
    return None


class ScientificLedger(RetrospectiveLedger):
    def __init__(self, task, store, gateway, budget, goal_feedback=True):
        super().__init__(task, store, gateway, None, budget)
        self.goal_feedback = goal_feedback

    def feedback(self, evaluation):
        if self.goal_feedback:
            return evaluation
        # Allow-list validity only. Goal scores, feasibility, rankings, and detailed
        # evaluator errors may reveal objective values and remain host-private.
        result = {k: evaluation[k] for k in ("valid", "structurally_evaluable") if k in evaluation}
        if not result.get("valid", False):
            result["error"] = "Design could not be evaluated; detailed feedback is withheld in this control."
        return result

    def state(self):
        result = super().state()
        for row in result["history"]:
            row["evaluation"] = self.feedback(row["evaluation"])
        return result

    def dispatch(self, action, payload):
        if action == "tool":
            if payload["name"] == "science.evaluate_design":
                raise ValueError("Use biosyn_submit so every scientific evaluation is saved and counted")
            if not self.goal_feedback and payload["name"] not in {
                "science.synthesize_circuit", "science.assemble_circuit", "science.assemble_mrna",
                "science.enumerate_pathways", "science.assemble_regulatory_design",
            }:
                raise ValueError("This tool has not been approved for the no-goal-feedback control")
        return super().dispatch(action, payload)

    def submit(self, candidate):
        if len(self.history) >= self.budget["candidate_evaluations"]:
            raise ValueError("Candidate evaluation budget exhausted")
        if not isinstance(candidate, dict) or set(candidate) != {"id", "parent_id", "design", "rationale"}:
            raise ValueError("Candidate requires id, parent_id, design and rationale")
        cid = candidate["id"]
        if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,59}", cid):
            raise ValueError("Invalid candidate id")
        ids = {r["candidate"]["id"] for r in self.history}
        if cid in ids or (candidate["parent_id"] is not None and candidate["parent_id"] not in ids):
            raise ValueError("Duplicate id or unknown parent")
        if not isinstance(candidate["design"], dict) or not isinstance(candidate["rationale"], str):
            raise ValueError("Invalid design or rationale type")
        candidate = json.loads(json.dumps(candidate, allow_nan=False))
        started = time.monotonic()
        self.store.event("candidate_received", candidate=candidate)
        try:
            receipt = self.gateway.call("science.evaluate_design", {"design": candidate["design"]})
            full_evaluation = receipt["data"]
            artifact = full_evaluation["artifact"]
            evaluation = {k: v for k, v in full_evaluation.items() if k not in ("artifact", "raw")}
            evaluation["structurally_evaluable"] = evaluation.get("valid", False)
            call_id = receipt["call_id"]
        except (ToolError, ValueError, TimeoutError) as exc:
            evaluation = {
                "valid": False,
                "structurally_evaluable": False,
                "goal_met": False,
                "error": f"{type(exc).__name__}: {exc}"[:3000],
            }
            artifact = candidate["design"]
            full_evaluation = evaluation
            call_id = None
        row = {
            "candidate": candidate,
            "evaluation": evaluation,
            "full_evaluation": full_evaluation,
            "artifact": artifact,
            "call_id": call_id,
            "seconds": time.monotonic() - started,
            "design_sha256": hashlib.sha256(json.dumps(artifact, sort_keys=True).encode()).hexdigest(),
        }
        self.history.append(row)
        self.store.write(f"candidate-{cid}.json", row)
        self.store.write("ledger.json", self.state())
        self.store.event("scientific_feedback", candidate_id=cid, evaluation=evaluation, call_id=call_id)
        return {
            "candidate_id": cid,
            **self.feedback(evaluation),
            "budget": {
                k: v for k, v in self.state().items() if k.startswith("remaining_") or k == "finalization_due"
            },
        }


def select_fixed(history, objectives):
    """Equal mean percentile rank across declared objectives, after hard validity checks."""
    valid = [r for r in history if r["evaluation"].get("valid")]
    if not valid:
        return history[0]
    reached = [r for r in valid if r["evaluation"].get("goal_met") is True]
    if reached:
        valid = reached
    feasible = [r for r in valid if r["evaluation"].get("feasible", True)]
    if feasible:
        valid = feasible
    if "truth_table_accuracy" in valid[0]["evaluation"]["metrics"]:
        best_accuracy = max(r["evaluation"]["metrics"]["truth_table_accuracy"] for r in valid)
        valid = [r for r in valid if r["evaluation"]["metrics"]["truth_table_accuracy"] == best_accuracy]

    def score(row):
        total = 0
        for obj in objectives:
            name, direction = obj["metric"], obj["direction"]
            v = row["evaluation"]["metrics"][name]
            values = [r["evaluation"]["metrics"][name] for r in valid]
            discrete = name.endswith("_count") or name in {"dna_length", "truth_table_accuracy"}
            total += sum(
                (x <= v if direction == "maximize" else x >= v)
                or (not discrete and math.isclose(x, v, rel_tol=1e-6, abs_tol=1e-6))
                for x in values
            ) / len(values)
        return total / len(objectives)

    return max(valid, key=lambda r: (score(r), -history.index(r)))


def fixed_search(task, resources, profile, ledger, seed):
    """A predefined, non-LLM search policy; it receives the same scientific feedback."""
    rng = random.Random(seed)
    kind = task["track"]
    if kind in ("bacterial_cds", "mammalian_mrna"):
        codons = resources["codons_by_amino_acid"]
        protein = task["inputs"]["protein_sequence"]
        stop = task["inputs"].get("required_terminal_stop", "TAA")
        weights = resources["codon_usage_counts"]
        design = {"cds": "".join(max(codons[a], key=lambda c: (weights[c], c)) for a in protein) + stop}
        if kind == "mammalian_mrna":
            design["assay_barcode"] = task["inputs"]["fixed_assay_barcode"]
    elif kind == "genetic_circuit":
        from biosyn.applications.computational_circuit import synthesize_nor

        graph = synthesize_nor(task["inputs"]["truth_table"])
        groups = {}
        for part, info in resources["gates"].items():
            groups.setdefault(info["group"], []).append(part)
        selected = rng.sample(list(groups), len(graph["gates"]))
        design = {
            "gates": [dict(g, part=rng.choice(groups[group])) for g, group in zip(graph["gates"], selected)],
            "output": graph["output"],
        }
    else:
        from biosyn.applications.computational_metabolic import initial_pathway

        design = initial_pathway(task, resources, rng)
    parent = None
    for index in range(profile["budget"]["candidate_evaluations"]):
        if index:
            chosen = select_fixed(ledger.history, profile["objectives"])
            parent = chosen["candidate"]["id"]
            design = copy.deepcopy(chosen["candidate"]["design"])
            if kind in ("bacterial_cds", "mammalian_mrna"):
                positions = list(range(len(protein)))
                # Half the proposals target the first 30 codons, half the complete CDS.
                pool = positions[:30] if index % 2 else positions
                n = min(len(pool), max(1, len(pool) // 10))
                chunks = [design["cds"][j : j + 3] for j in range(0, len(design["cds"]), 3)]
                for pos in rng.sample(pool, n):
                    alternatives = [c for c in codons[protein[pos]] if c != chunks[pos]]
                    if alternatives:
                        chunks[pos] = rng.choices(
                            alternatives, weights=[weights[c] for c in alternatives], k=1
                        )[0]
                design["cds"] = "".join(chunks)
            elif kind == "genetic_circuit":
                gates = design["gates"]
                if index % 2 and len(gates) > 1:
                    a, b = rng.sample(range(len(gates)), 2)
                    gates[a]["part"], gates[b]["part"] = gates[b]["part"], gates[a]["part"]
                else:
                    a = rng.randrange(len(gates))
                    used = {resources["gates"][g["part"]]["group"] for j, g in enumerate(gates) if j != a}
                    options = [p for p, v in resources["gates"].items() if v["group"] not in used]
                    gates[a]["part"] = rng.choice(options)
            else:
                from biosyn.applications.computational_metabolic import mutate_pathway

                design = mutate_pathway(design, task, resources, rng, index)
        ledger.submit(
            {
                "id": f"fixed_{index:02d}",
                "parent_id": parent,
                "design": design,
                "rationale": "Predeclared simple search; no LLM controller.",
            }
        )
    selected = select_fixed(ledger.history, profile["objectives"])
    ledger.dispatch(
        "finalize",
        {"candidate_id": selected["candidate"]["id"], "selection_source": "predeclared_simple_search"},
    )


def run_episode(root, dataset, out, row, method, seed, endpoint, key):
    case = out / f"{row['case_id']}--{method}--{seed}"
    case.mkdir(parents=True, exist_ok=False)
    public, work = case / "public", case / "work"
    public.mkdir()
    work.mkdir()
    agent_source = case / "agent-source"
    if method != "biomni_upstream":
        prepare_source(
            Path(__file__).resolve().parents[3],
            agent_source,
            expose_biosyn_helpers=method != "biomni_native",
        )
    else:
        runner = Path(__file__).with_name("biomni_upstream_worker.py")
        hashes = {"runner/biomni_upstream_worker.py": sha(runner)}
        hashes.update({
            str(path.relative_to(root)): sha(path)
            for path in sorted((root / "Biomni/biomni").rglob("*.py"))
        })
        write(case / "agent-source-hashes.json", hashes)
    task = json.loads((dataset / row["task_path"]).read_text())
    profile = json.loads((dataset / row["profile_path"]).read_text())
    resources = json.loads((dataset / row["resources_path"]).read_text())
    resource_receipts = copy_public_resources(task, dataset, public)
    budget = dict(BUDGET, **profile["budget"])
    budget["llm_request_timeout"] = llm_request_timeout(method)
    config = llm_config()
    # RPC (360s) outlasts the MCP evaluator (300s); REPL (420s) outlasts the RPC.
    config["code_timeout"] = 420
    config["recursion_limit"] = budget.get("recursion_limit", config["recursion_limit"])
    config["llm"]["timeout"] = llm_request_timeout(method)
    config["llm"]["extra_body"]["seed"] = seed
    config["llm"]["max_tokens"] = budget["max_output_tokens"]
    if method == "biomni_native":
        config.update(
            biomni_tool_modules=BIOMNI_NATIVE_MODULES,
            biomni_excluded_tools=BIOMNI_EXCLUDED_TOOLS,
            preserve_native_biomni_knowhow=True,
            register_runtime_tools=False,
        )
    write(public / "task.json", dict(task, id=task["case_id"]))
    write(public / "config.json", config)
    write(public / "scientific-profile.json", dict(profile, method=method, seed=seed))
    write(public / "design-resources.json", resources)
    settings = ServerConfig(
        command=[
            str(root / ".venv-design-eval/bin/python"),
            "-m",
            "biosyn.mcp.computational",
            "--task",
            str(public / "task.json"),
            "--resources",
            str(root / "resources/application_suite_v3"),
            "--catalog",
            str(dataset / row["catalog_path"]),
            "--resource-lock",
            str(dataset / row["resource_lock_path"]),
            "--root",
            str(root),
        ],
        env={
            "PYTHONPATH": str(Path(__file__).resolve().parents[2]),
            "OMP_NUM_THREADS": "2",
            "OPENBLAS_NUM_THREADS": "2",
            "TF_CPP_MIN_LOG_LEVEL": "2",
            "TF_ENABLE_ONEDNN_OPTS": "0",
            "CUDA_VISIBLE_DEVICES": "-1",
        },
        timeout=300,
    )
    store = RunStore(case / "ledger")
    start = time.monotonic()
    result = {
        "case_id": row["case_id"],
        "method": method,
        "seed": seed,
        "model": None if method == "fixed_search" else MODEL,
        "task_sha256": sha(public / "task.json"),
        "base_profile_sha256": sha(dataset / row["profile_path"]),
        "resources_sha256": sha(public / "design-resources.json"),
        "budget": budget,
        "experimental_outcome": "not_measured",
        "execution_host": platform.node(),
        "controller_source": str(Path(__file__).resolve().parents[2]),
        "scientific_source": settings.env["PYTHONPATH"],
        "public_resources": resource_receipts,
        "worker_request_timeout_seconds": config["llm"]["timeout"],
        "proxy_request_timeout_seconds": llm_request_timeout(method),
        "episode_wall_deadline_seconds": budget["wall_seconds"],
    }
    with ToolGateway({"science": settings}, store, max_calls=budget["mcp_calls"]) as gateway:
        write(public / "catalog.json", {} if method == "biomni_upstream" else {
            k: v for k, v in gateway.catalog.items() if k != "science.evaluate_design"
        })
        ledger = ScientificLedger(task, store, gateway, budget,
                                  goal_feedback=method != "biosyn_no_goal_feedback")
        if method == "fixed_search":
            fixed_search(task, resources, dict(profile, budget=budget), ledger, seed)
            result.update(process_returncode=0, timeout=False, api_usage={"calls": 0, "total_tokens": 0})
        elif method == "llm_independent":
            from biosyn.applications.independent_candidates import run_independent
            result.update(run_independent(root, case, public, budget, ledger, seed, endpoint, key,
                                          lambda history: select_fixed(history, profile["objectives"]),
                                          agent_source))
        else:
            with tempfile.TemporaryDirectory(prefix="biosyn-compare-") as temp:
                llm_sock, eval_sock = Path(temp) / "llm.sock", Path(temp) / "eval.sock"
                with (
                    chat_proxy(
                        llm_sock,
                        key,
                        case / "api-usage.jsonl",
                        max_calls=1 if method == "llm_direct" else budget["llm_requests"],
                        max_tokens=budget["max_output_tokens"],
                        max_total_tokens=budget["model_tokens"],
                        model=MODEL,
                        endpoint=endpoint["base_url"].rstrip("/") + "/chat/completions",
                        timeout=llm_request_timeout(method),
                    ) as usage,
                    ledger_server(eval_sock, ledger) if method != "biomni_upstream" else nullcontext(),
                ):
                    ledger.llm_usage = usage
                    if method == "biomni_upstream":
                        runner = Path(__file__).with_name("biomni_upstream_worker.py")
                        cmd = upstream_biomni_command(root, public, work, llm_sock, runner)
                    else:
                        cmd = command(root, public, work, llm_sock, agent_source)
                        cmd[-1] = "biosyn.applications.comparison_worker"
                    if method in {"biosyn", "biosyn_no_goal_feedback", "biomni_native"}:
                        cmd[1:1] = ["--ro-bind", str(eval_sock), "/proxy/eval.sock"]
                    write(
                        case / "execution.json",
                        {
                            "command": cmd,
                            "feedback_access": method in {
                                "biosyn", "biosyn_no_goal_feedback", "biomni_native"
                            },
                            "goal_feedback_access": method in {"biosyn", "biomni_native"},
                            "biosyn_scientific_mcp_access": method not in {
                                "biomni_native", "biomni_upstream"
                            },
                            "native_biomni_tool_modules": (
                                BIOMNI_NATIVE_MODULES if method == "biomni_native" else []
                            ),
                            "excluded_native_biomni_tools": (
                                BIOMNI_EXCLUDED_TOOLS if method == "biomni_native" else []
                            ),
                            "upstream_biomni_only": method == "biomni_upstream",
                            "native_biomni_tool_policy": (
                                "all_upstream_modules" if method == "biomni_upstream" else "declared_subset"
                            ),
                            "biosyn_source_mounted": method != "biomni_upstream",
                            "direct_design_protocol": (
                                "upstream Biomni tools; no evaluator feedback; one post-exit score"
                                if method == "biomni_upstream" else
                                "one LLM call, one Python construction, one proposal, no tool feedback"
                            ),
                            "private_labels_mounted": False,
                        },
                    )
                    with (case / "agent.log").open("w") as log:
                        proc = subprocess.Popen(
                            cmd,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
                        )
                        try:
                            result["process_returncode"] = proc.wait(timeout=budget["wall_seconds"])
                            result["timeout"] = False
                        except subprocess.TimeoutExpired:
                            os.killpg(proc.pid, signal.SIGKILL)
                            result.update(process_returncode=proc.wait(), timeout=True)
                    result["api_usage"] = dict(usage)
            worker_failure = work / "output/worker-failure.json"
            if worker_failure.is_file():
                result["worker_failure"] = json.loads(worker_failure.read_text())
                result["failure_category"] = result["worker_failure"]["failure_category"]
            if result.get("timeout"):
                result["failure_category"] = "episode_timeout_requires_review"
            proposal = work / "output/direct-design.json"
            if method == "llm_direct" and proposal.exists() and result["process_returncode"] == 0:
                ledger.submit(
                    {
                        "id": "direct",
                        "parent_id": None,
                        "design": json.loads(proposal.read_text()),
                        "rationale": "One-shot LLM proposal; scored only after the model finished.",
                    }
                )
                ledger.dispatch(
                    "finalize", {"candidate_id": "direct", "selection_source": "single_direct_proposal"}
                )
            upstream_proposal = work / "output/proposed-design.json"
            if (method == "biomni_upstream" and upstream_proposal.exists()
                    and result["process_returncode"] == 0):
                from biosyn.applications.comparison_protocol import unpack_design
                design, form = unpack_design(
                    json.loads(upstream_proposal.read_text()), profile["decision_schema"]
                )
                write(work / "output/submission-contract.json", {
                    "version": "comparison-2.0.0", "input_form": form,
                    "biological_fields_modified": False,
                })
                ledger.submit({
                    "id": "biomni_upstream_final", "parent_id": None, "design": design,
                    "rationale": "Upstream Biomni final proposal; evaluated once after agent exit.",
                })
                ledger.dispatch("finalize", {
                    "candidate_id": "biomni_upstream_final",
                    "selection_source": "post_exit_single_evaluation",
                })
        result.update(final=ledger.final, candidate_count=len(ledger.history), mcp_calls=gateway.calls)
        if ledger.final:
            selected = json.loads((store.path / "final.json").read_text())
            write(case / "submission.json", selected["artifact"])
            write(case / "evaluation.json", selected["full_evaluation"])
            result["design_valid"] = selected["evaluation"].get("valid", False)
            result["goal_met"] = selected["evaluation"].get("goal_met")
        result["workflow_complete"] = bool(
            ledger.final and result["process_returncode"] == 0 and not result["timeout"]
        )
    if ledger.final and result.get("design_valid"):
        rescored = rescore_final(settings, case, selected["artifact"], selected["full_evaluation"])
        result["independent_rescore"] = {k: v for k, v in rescored.items() if k != "evaluation"}
    else:
        result["independent_rescore"] = {
            "completed": False,
            "consistent": False,
            "reason": "No valid final design",
        }
    result["seconds"] = time.monotonic() - start
    write(case / "result.json", result)
    return result


def execute_record(root, dataset, out, row, method, seed, endpoint, key):
    failed = out / f"{row['case_id']}--{method}--{seed}"
    if failed.exists():
        if (failed / "result.json").exists():
            return json.loads((failed / "result.json").read_text())
        # Preserve interrupted attempts; restarting the experiment never silently gives extra attempts.
        result = {
            "case_id": row["case_id"],
            "method": method,
            "seed": seed,
            "workflow_complete": False,
            "design_valid": False,
            "exception": "Interrupted episode: files existed without result.json",
            "failure_category": "infrastructure_interruption",
        }
        write(failed / "result.json", result)
        return result
    try:
        return run_episode(root, dataset, out, row, method, seed, endpoint, key)
    except Exception as exc:
        import traceback

        result = {
            "case_id": row["case_id"],
            "method": method,
            "seed": seed,
            "workflow_complete": False,
            "design_valid": False,
            "exception": f"{type(exc).__name__}: {exc}",
            "failure_category": "execution_exception_requires_review",
        }
        failed.mkdir(exist_ok=True)
        (failed / "failure-traceback.txt").write_text(traceback.format_exc())
        write(failed / "result.json", result)
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--endpoint", type=Path)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["llm_direct", "llm_independent", "fixed_search", "biomni_native", "biomni_upstream", "biosyn", "biosyn_no_goal_feedback"],
        default=["llm_direct", "biosyn"],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--case-ids", nargs="*")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--execute-methods",
        nargs="+",
        choices=["llm_direct", "llm_independent", "fixed_search", "biomni_native", "biomni_upstream", "biosyn", "biosyn_no_goal_feedback"],
        help="Execute a subset while preserving the full declared schedule; resume runs the rest",
    )
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Use 1..4 workers to respect scientific-model CPU and memory allocations")
    if len(set(args.methods)) != len(args.methods) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate methods or seeds")
    root, dataset, out = args.root.resolve(), args.dataset.resolve(), args.output.resolve()
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["status"] != "ready":
        raise ValueError("Comparison dataset must pass scientific and protocol checks before execution")
    for item in manifest["files"]:
        if sha(dataset / item["path"]) != item["sha256"]:
            raise ValueError("Dataset file changed: " + item["path"])
    execute_methods = args.execute_methods or args.methods
    if set(execute_methods) - set(args.methods):
        raise ValueError("Execution subset must belong to the declared methods")
    if any(m != "fixed_search" for m in execute_methods):
        if not args.endpoint:
            raise ValueError("LLM methods require an endpoint")
        endpoint = json.loads(args.endpoint.read_text())
        key = dotenv_values(endpoint["env_file"])["BIOSYN_GLM_API_KEY"]
    else:
        endpoint, key = {}, ""
    cases = [r for r in manifest["cases"] if not args.case_ids or r["case_id"] in args.case_ids]
    if not cases or args.case_ids and set(args.case_ids) - {r["case_id"] for r in cases}:
        raise ValueError("Unknown or empty case selection")
    schedule = [(r, m, s) for r in cases for s in args.seeds for m in args.methods]
    random.Random(20261028).shuffle(schedule)
    records = [{"case_id": r["case_id"], "method": m, "seed": s} for r, m, s in schedule]
    hashes = {str(p.relative_to(root)): sha(p) for p in Path(__file__).resolve().parents[1].rglob("*.py")}
    experiment = {
        "dataset_manifest_sha256": sha(dataset / "manifest.json"),
        "expected_episodes": len(records),
        "methods": args.methods,
        "seeds": args.seeds,
        "case_ids": [r["case_id"] for r in cases],
    }
    if args.resume:
        for name, expected in [
            ("schedule.json", records),
            ("source-hashes.json", hashes),
            ("experiment.json", experiment),
        ]:
            if json.loads((out / name).read_text()) != expected:
                raise ValueError("Cannot resume after experiment or code change: " + name)
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out / "schedule.json", records)
        write(out / "source-hashes.json", hashes)
        write(out / "experiment.json", experiment)
        for relative in hashes:
            target = out / "source-snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / relative).read_bytes())
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {
            pool.submit(execute_record, root, dataset, out, r, m, seed, endpoint, key): (
                r["case_id"],
                m,
                seed,
            )
            for r, m, seed in schedule
            if m in execute_methods
        }
        for future in as_completed(pending):
            identity = pending[future]
            result = future.result()
            results[identity] = result
            ordered = [
                results[(r["case_id"], m, seed)]
                for r, m, seed in schedule
                if (r["case_id"], m, seed) in results
            ]
            write(out / "results.json", ordered)
            print(
                json.dumps(
                    {
                        "completed": len(results),
                        "expected": len(schedule),
                        "case_id": identity[0],
                        "method": identity[1],
                        "seed": identity[2],
                        "workflow_complete": result.get("workflow_complete"),
                        "design_valid": result.get("design_valid"),
                        "seconds": result.get("seconds"),
                        "exception": result.get("exception"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
