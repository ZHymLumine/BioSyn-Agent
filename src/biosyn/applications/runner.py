"""Biomni sandbox + host-owned evaluation ledger + real MCP scientific service."""
import argparse
import hashlib
import json
import os
import re
import signal
import socketserver
import subprocess
import tempfile
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler
from pathlib import Path

import jsonschema
from dotenv import dotenv_values

from biosyn.agent.state import RunStore
from biosyn.benchmarks.proxy import deepseek_proxy
from biosyn.benchmarks.runner import command, write
from biosyn.config import ServerConfig
from biosyn.mcp.client import ToolGateway


class Ledger:
    def __init__(self, task, profile, gateway, store):
        self.task, self.profile, self.gateway, self.store = task, profile, gateway, store
        self.history = []
        self.final = None
        self.lock = threading.RLock()
        self.llm_usage = {"calls": 0, "total_tokens": 0}

    def state(self):
        return {"history": self.history, "final": self.final,
                "remaining_evaluations": self.task["budget"]["candidate_evaluations"] - len(self.history),
                "remaining_llm_requests": max(0, self.task["budget"].get("llm_requests", 0) - self.llm_usage["calls"]),
                "remaining_model_tokens": max(0, self.task["budget"].get("model_tokens", 0) - self.llm_usage["total_tokens"])}

    def dispatch(self, action, payload):
        with self.lock:
            if action == "state":
                return self.state()
            if self.final is not None:
                raise ValueError("Finalized; no more evaluations or selection changes")
            if action == "submit":
                return self.submit(payload["candidate"])
            if action == "finalize":
                return self.finalize(payload["candidate_id"])
            raise ValueError("Unknown ledger operation")

    def submit(self, candidate):
        if len(self.history) >= self.task["budget"]["candidate_evaluations"]:
            raise ValueError("Candidate evaluation budget exhausted")
        if set(candidate) != {"id", "parent_id", "design", "rationale"}:
            raise ValueError("Candidate requires exactly id, parent_id, design, rationale")
        cid = candidate["id"]
        if not isinstance(cid, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,59}", cid):
            raise ValueError("Invalid candidate id")
        ids = {row["candidate"]["id"] for row in self.history}
        if cid in ids or (candidate["parent_id"] is not None and candidate["parent_id"] not in ids):
            raise ValueError("Duplicate id or unknown parent")
        if not isinstance(candidate["design"], dict) or not isinstance(candidate["rationale"], str):
            raise ValueError("design must be an object; rationale must be a string")
        candidate = json.loads(json.dumps(candidate, allow_nan=False))
        start = time.monotonic()
        self.store.event("candidate_received", candidate=candidate)
        try:
            receipt = self.gateway.call("application.evaluate_design", {"design": candidate["design"]})
            evaluation = receipt["data"]
            call_id = receipt["call_id"]
        except Exception as exc:
            evaluation = {"valid": False, "feasible": False, "error": f"{type(exc).__name__}: {exc}"}
            call_id = None
        row = {"candidate": candidate, "evaluation": evaluation, "call_id": call_id,
               "elapsed_seconds": time.monotonic()-start,
               "design_sha256": hashlib.sha256(json.dumps(candidate["design"], sort_keys=True).encode()).hexdigest()}
        self.history.append(row)
        self.store.write(f"candidate-{cid}.json", row)
        self.store.write("ledger.json", self.state())
        self.store.event("feedback_returned", **row)
        return row

    def finalize(self, cid):
        row = next((x for x in self.history if x["candidate"]["id"] == cid), None)
        if row is None or not row["evaluation"].get("valid"):
            raise ValueError("Finalize requires an existing scientifically evaluable design")
        self.final = {"candidate_id": cid, "design_sha256": row["design_sha256"],
                      "feasible": row["evaluation"]["feasible"]}
        self.store.write("final.json", row)
        self.store.write("ledger.json", self.state())
        self.store.event("finalized", **self.final)
        return self.final


@contextmanager
def ledger_server(path, ledger):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            size = int(self.headers.get("Content-Length", "0"))
            try:
                if not 0 < size <= 100000:
                    raise ValueError("Payload size outside bounds")
                data = json.loads(self.rfile.read(size))
                result = ledger.dispatch(self.path.removeprefix("/"), data)
                body, status = json.dumps(result, allow_nan=False).encode(), 200
            except Exception as exc:
                body, status = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode(), 400
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = socketserver.UnixStreamServer(str(path), Handler)
    os.chmod(path, 0o600)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def checked_bindings(root, task):
    schema = json.loads((root / "tasks/application_suite_v3/task.schema.json").read_text())
    jsonschema.validate(task, schema)
    if task["status"] != "ready":
        raise ValueError("Only bound ready tasks can execute")

    def walk(value):
        if isinstance(value, dict):
            if set(value) == {"path", "sha256"}:
                path = (root / value["path"]).resolve()
                if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest() != value["sha256"]:
                    raise ValueError("Task resource/profile changed since binding")
            for item in value.values():
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    walk(task)


def run(root, kind, destination):
    root = root.resolve()
    task = json.loads((root / f"tasks/application_suite_v3/cases/{kind}/task.json").read_text())
    checked_bindings(root, task)
    profile_path = root / task["context"]["profile"]["path"]
    profile = json.loads(profile_path.read_text())
    case = destination.resolve()
    case.mkdir(parents=True, exist_ok=False)
    public, work = case / "public", case / "work"
    public.mkdir()
    work.mkdir()
    write(public / "task.json", task)
    write(public / "profile.json", profile)
    (public / "protocol.md").write_bytes((root / "tasks/application_suite_v3/protocol.md").read_bytes())
    store = RunStore(case / "ledger")
    settings = ServerConfig(command=[str(root / ".venv-application-tools/bin/python"), "-m",
        "biosyn.mcp.application", "--profile", str(profile_path), "--resources", str(root / "resources/application_suite_v3")],
        env={"PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
             "TF_CPP_MIN_LOG_LEVEL": "2", "TF_ENABLE_ONEDNN_OPTS": "0"}, timeout=180)
    api_key = dotenv_values(root / ".secrets/deepseek.env")["DEEPSEEK_API_KEY"]
    started = time.monotonic()
    with ToolGateway({"application": settings}, store, max_calls=26) as gateway:
        # The public catalogue contains no measured labels or optimized answers.
        catalog = gateway.call("application.describe_resources", {})["data"]
        write(public / "catalog.json", catalog)
        ledger = Ledger(task, profile, gateway, store)
        with tempfile.TemporaryDirectory(prefix="biosyn-v3-") as temp:
            llm_socket, eval_socket = Path(temp) / "llm.sock", Path(temp) / "eval.sock"
            with deepseek_proxy(llm_socket, api_key, case / "api-usage.jsonl",
                                max_calls=task["budget"]["llm_requests"], max_tokens=4096,
                                max_total_tokens=task["budget"]["model_tokens"]) as usage, ledger_server(eval_socket, ledger):
                ledger.llm_usage = usage
                cmd = command(root, public, work, llm_socket)
                cmd[-1] = "biosyn.applications.worker"
                cmd[1:1] = ["--ro-bind", str(eval_socket), "/proxy/eval.sock"]
                write(case / "execution.json", {"command": cmd, "budget": task["budget"],
                    "profile_sha256": hashlib.sha256(profile_path.read_bytes()).hexdigest(),
                    "token_budget_enforcement": "Reject next request after observed cumulative API usage reaches limit; one request may cross it."})
                with (case / "agent.log").open("w") as log:
                    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
                    timed_out = False
                    try:
                        code = proc.wait(timeout=task["budget"]["wall_seconds"])
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        code = proc.wait()
                        timed_out = True
                result = {"process_returncode": code, "timeout": timed_out, "api_usage": dict(usage),
                          "final": ledger.final, "candidate_count": len(ledger.history)}
    result["elapsed_seconds"] = time.monotonic() - started
    write(case / "result.json", result)
    # Fresh evaluator process, no model memory or state from the agent episode.
    verification = subprocess.run([str(root / ".venv-application-tools/bin/python"), "-m",
        "biosyn.applications.verify", "--case", str(case), "--profile", str(profile_path),
        "--resources", str(root / "resources/application_suite_v3")],
        env={**os.environ, "PYTHONPATH": str(root / "src"), "OMP_NUM_THREADS": "2",
             "TF_CPP_MIN_LOG_LEVEL": "2", "TF_ENABLE_ONEDNN_OPTS": "0"}, capture_output=True, text=True, timeout=300)
    (case / "verification.log").write_text(verification.stdout + verification.stderr)
    result["verification_returncode"] = verification.returncode
    write(case / "result.json", result)
    print(json.dumps(result))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=["bacterial", "mammalian", "circuit", "metabolic"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.root, args.kind, args.output)


if __name__ == "__main__":
    main()
