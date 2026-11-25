"""Independent one-shot candidates in separate namespaces, with host-only selection."""

import copy
import json
import os
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path

from biosyn.applications.retrospective import MODEL
from biosyn.benchmarks.proxy import chat_proxy
from biosyn.benchmarks.runner import write
from biosyn.applications.comparison_sandbox import command


def stage_candidate(public, destination, seed, index):
    shutil.copytree(public, destination)
    config_path = destination / "config.json"
    config = json.loads(config_path.read_text())
    # A deterministic stream per episode; each request still has the same task and messages.
    config["llm"]["extra_body"]["seed"] = seed * 1000 + index
    write(config_path, config)
    return config["llm"]["extra_body"]["seed"]


def run_independent(
    root,
    case,
    public,
    budget,
    ledger,
    seed,
    endpoint,
    key,
    select,
    agent_source,
):
    started = time.monotonic()
    deadline = started + budget["wall_seconds"]
    generation_deadline = deadline - budget["finalization_wall_reserve"]
    attempts, proposals = [], []
    with tempfile.TemporaryDirectory(prefix="biosyn-independent-") as temp:
        socket = Path(temp) / "llm.sock"
        # Independent generations have no client-side request deadline. The
        # process wait below enforces the declared episode generation deadline.
        with chat_proxy(socket, key, case / "api-usage.jsonl",
                        max_calls=min(budget["candidate_evaluations"], budget["llm_requests"]),
                        max_tokens=budget["max_output_tokens"],
                        max_total_tokens=budget["model_tokens"], model=MODEL,
                        endpoint=endpoint["base_url"].rstrip("/") + "/chat/completions",
                        timeout=None) as usage:
            ledger.llm_usage = usage
            for index in range(budget["candidate_evaluations"]):
                if (time.monotonic() >= generation_deadline
                        or usage["calls"] >= budget["llm_requests"]
                        or usage["total_tokens"] >= budget["model_tokens"]):
                    break
                candidate_dir = case / "independent" / f"proposal-{index:02d}"
                candidate_dir.mkdir(parents=True)
                candidate_public = candidate_dir / "public"
                candidate_seed = stage_candidate(public, candidate_public, seed, index)
                work = candidate_dir / "work"
                work.mkdir()
                cmd = command(root, candidate_public, work, socket, agent_source)
                cmd[-1] = "biosyn.applications.comparison_worker"
                attempt = {"index": index, "request_seed": candidate_seed,
                           "evaluation_socket_mounted": False, "other_candidate_files_mounted": False,
                           "command": cmd}
                with (candidate_dir / "agent.log").open("w") as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                            start_new_session=True,
                                            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
                    try:
                        attempt["returncode"] = proc.wait(timeout=max(1, generation_deadline-time.monotonic()))
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        attempt.update(returncode=proc.wait(), generation_budget_exhausted=True)
                failure = work / "output/worker-failure.json"
                if failure.exists():
                    attempt["worker_failure"] = json.loads(failure.read_text())
                proposal = work / "output/direct-design.json"
                if attempt["returncode"] == 0 and proposal.exists():
                    proposals.append({"id": f"independent_{index:02d}", "parent_id": None,
                                      "design": json.loads(proposal.read_text()),
                                      "rationale": "Independent one-shot proposal; no evaluation or other proposals observed."})
                    attempt["proposal_saved"] = True
                else:
                    attempt["proposal_saved"] = False
                attempts.append(attempt)
                write(candidate_dir / "attempt.json", attempt)
            usage_result = copy.deepcopy(usage)
    # No generation process remains when scoring begins. The selector sees only the same public metrics.
    write(case / "independent-generation.json", {"attempts": attempts,
          "generation_completed_before_evaluation": True, "proposals": len(proposals)})
    unscored = []
    for proposal in proposals:
        if deadline - time.monotonic() < 300:
            unscored.append(proposal["id"])
            continue
        ledger.submit(proposal)
    if ledger.history:
        selected = select(ledger.history)
        ledger.dispatch("finalize", {"candidate_id": selected["candidate"]["id"],
                                      "selection_source": "predeclared_independent_candidate_selector"})
    return {"process_returncode": 0 if ledger.final else 1, "timeout": False,
            "api_usage": usage_result, "generation_attempts": len(attempts),
            "constructed_proposals": len(proposals), "unscored_due_to_budget": unscored,
            "candidate_failures": sum(not a["proposal_saved"] for a in attempts),
            "independent_request_timeout_seconds": None,
            "independent_worker_request_timeout_seconds": None,
            "independent_proxy_request_timeout_seconds": None,
            "independent_generation_deadline_enforced": True,
            "independent_protocol": (
                "separate namespaces; worker and proxy request timeouts disabled; "
                "episode generation deadline enforced; "
                "shared total budgets; score only after all generation"
            )}
