"""One Biomni episode, run inside an empty Bubblewrap filesystem and network namespace."""
import hashlib
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

from pydantic import BaseModel

from biosyn.agent.planner import BiomniPlanner
from biosyn.agent.state import RunStore
from biosyn.benchmarks.contracts import instructions, validate_json
from biosyn.config import RuntimeConfig
from biosyn.mcp.client import ToolGateway


class EpisodeTask(BaseModel):
    id: str
    request: str


class Episode:
    def __init__(self, packet, config):
        self.packet = packet
        self.config = config
        self.store = RunStore(Path("/work/output"))
        self.gateway = ToolGateway(config.servers, self.store, max_calls=60)
        self.lock = threading.Lock()
        self.submission = None
        task = packet["task"]
        request = ("Solve the current task in /public/task.json. Read /public/protocol.md and the other public "
                   "resources in /public when relevant. Inspect large JSON files selectively in Python; "
                   "do not dump full catalogues or long sequences into the conversation. "
                   "Do not use any other task or prior answers. "
                   + instructions(task["task_type"], packet["answer_format"]))
        if task["task_type"] == "E1":
            request += (" No validated expression foundation model is configured for this smoke test. "
                        "Use public sequences only, state this limitation, and do not invent model scores.")
        self.state = SimpleNamespace(run_id=task["id"], task=EpisodeTask(id=task["id"], request=request))
        self.store.write("episode.json", packet)

    def summary(self):
        return {"submitted": self.submission is not None, "remaining_mcp_calls": 60 - self.gateway.calls,
                "grading": "Private grade unavailable during this episode; final submission ends it."}

    def submit(self, candidate):
        with self.lock:
            if self.submission is not None:
                raise ValueError("Episode already finalized; no further submissions allowed")
            try:
                if self.packet["answer_format"] == "python":
                    source = Path(candidate["program_path"]).resolve(strict=True)
                    if not source.is_relative_to("/work"):
                        raise ValueError("Program must be saved in /work")
                    program = self.store.path / "submission.py"
                    program.write_bytes(source.read_bytes())
                    artifact = self.store.path / "submission.nt"
                    artifact.unlink(missing_ok=True)
                    proc = subprocess.run([sys.executable, str(program)], cwd="/work",
                        env={**os.environ, "BIOSYN_OUTPUT": str(artifact)}, capture_output=True, text=True, timeout=60)
                    self.store.write("program-execution.json", {"returncode": proc.returncode,
                        "stdout": proc.stdout[-20000:], "stderr": proc.stderr[-20000:]})
                    if proc.returncode or not artifact.is_file():
                        raise ValueError(f"Submitted program failed or omitted artifact: {proc.stderr[-5000:]}")
                    import sbol3
                    doc = sbol3.Document()
                    doc.read(str(artifact), sbol3.SORTED_NTRIPLES)
                    errors = doc.validate().errors
                    if errors:
                        raise ValueError(f"Invalid SBOL: {errors}")
                    result = {"contract_valid": True, "scope": "program execution and SBOL validation only",
                              "task_success": None}
                else:
                    source = Path(candidate["path"]).resolve(strict=True)
                    if not source.is_relative_to("/work"):
                        raise ValueError("JSON submission must be saved in /work")
                    value = json.loads(source.read_text())
                    catalog_path = Path("/public/expression_catalog.json")
                    result = validate_json(self.packet["task"], value,
                        json.loads(catalog_path.read_text()) if catalog_path.exists() else None)
                    artifact = self.store.path / "submission.json"
                    self.store.write("submission.json", value)
                self.submission = {**result, "artifact": artifact.name,
                    "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest()}
                self.store.write("submission-receipt.json", self.submission)
                self.store.event("submission_accepted", **self.submission)
                return self.submission
            except Exception as exc:
                self.store.event("submission_rejected", error=f"{type(exc).__name__}: {exc}")
                raise

    def run(self):
        status, error = "running", None
        self.store.event("episode_started")
        # These paths must remain unavailable inside the actual agent namespace.
        root = Path(self.packet["server_root"])
        private = [root / "tasks/circuit_reasoning/labels", root / "tasks/application_suite_v2/labels",
                   root / ".secrets/deepseek.env"]
        assert not any(p.exists() for p in private), "Isolation failed"
        self.store.write("isolation.json", {"private_paths_absent": True, "network_namespace": True,
                                          "llm_transport": "restricted Unix socket proxy"})
        try:
            with self.gateway:
                schema = {"type": "object", "properties": {"path": {"type": "string"},
                          "program_path": {"type": "string"}}, "additionalProperties": False}
                planner = BiomniPlanner(self.config, self, submission_schema=schema,
                    submission_description="Finalize a submission after public format checks; no private grade is exposed.",
                    instructions=("You are BioSyn-Agent running the existing Biomni A1 planner on one independent "
                        "task episode. Read the public task and contract. Use Python and registered tools to solve it. "
                        "Do not fabricate scores. Save the required artifact and call biosyn_submit exactly once "
                        "successfully; you may correct public format/execution errors before acceptance. After it "
                        "accepts, finish immediately. Acceptance means valid submission, not biological or benchmark success."))
                for round_number in (1, 2):
                    answer = planner.run_round(round_number, stop_when=lambda: self.submission is not None)
                    self.store.write(f"answer-{round_number}.json", {"answer": answer})
                    if self.submission:
                        break
                status = "submitted" if self.submission else "no_submission"
        except Exception as exc:
            status, error = "failed", f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.store.event("episode_finished", status=status, error=error)
            self.store.write("episode-result.json", {"status": status, "error": error, "submission": self.submission})


def main():
    packet = json.loads(Path("/public/packet.json").read_text())
    config = RuntimeConfig.model_validate(json.loads(Path("/public/config.json").read_text()))
    Episode(packet, config).run()


if __name__ == "__main__":
    main()
