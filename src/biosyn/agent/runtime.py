import platform
import subprocess
import threading
from importlib.metadata import version
from pathlib import Path

from biosyn.agent.evaluator import evaluate
from biosyn.agent.optimizer import select
from biosyn.agent.state import Candidate, DesignState, RunStore
from biosyn.mcp.client import ToolGateway


class DesignRuntime:
    def __init__(self, task, config, run_dir: Path, mode="biomni"):
        self.config = config
        self.store = RunStore(run_dir)
        self.state = DesignState(run_id=run_dir.name, task=task, mode=mode)
        self._lock = threading.RLock()
        self.gateway = ToolGateway(config.servers, self.store, task.max_tool_calls)
        self.store.write("task.json", task.model_dump(mode="json"))
        # Values in env/headers can contain credentials: record only their field names.
        public = config.model_dump(mode="json")
        for server in public["servers"].values():
            server["env"] = list(server["env"])
            server["headers_env"] = list(server["headers_env"])
        self.store.write("config.json", public)
        self.store.snapshot(self.state)
        self.store.event("run_started", mode=mode, task_id=task.id)

    def summary(self):
        with self._lock:
            return {"status": self.state.status, "round": self.state.round,
                    "remaining_tool_calls": max(0, self.state.task.max_tool_calls - self.gateway.calls),
                    "remaining_candidates": self.state.task.max_candidates - len(self.state.candidates),
                    "candidates": [c.model_dump() for c in self.state.candidates],
                    "evaluations": {k: v.model_dump() for k, v in self.state.evaluations.items()},
                    "best_id": self.state.best_id, "pareto_ids": self.state.pareto_ids}

    def submit(self, candidate: dict):
        with self._lock:
            if self.state.status != "running":
                raise ValueError("Run is no longer accepting candidates")
            candidate = Candidate.model_validate(candidate)
            if len(self.state.candidates) >= self.state.task.max_candidates:
                raise ValueError("Candidate budget exhausted")
            ids = {c.id for c in self.state.candidates}
            if candidate.id in ids:
                raise ValueError("Candidate ids are immutable; submit a new id for a revision")
            if candidate.parent_id is not None and candidate.parent_id not in ids:
                raise ValueError("parent_id must refer to an existing candidate")
            self.state.candidates.append(candidate)
            self.store.event("candidate_submitted", candidate=candidate.model_dump())
            self.store.snapshot(self.state)
            result = evaluate(candidate, self.state.task, self.gateway)
            self.state.evaluations[candidate.id] = result
            self.state.best_id, self.state.pareto_ids = select(self.state.evaluations, self.state.task.objectives)
            self.store.event("candidate_evaluated", evaluation=result.model_dump())
            self.store.snapshot(self.state)
            return result.model_dump()

    def run(self, planner_factory=None):
        try:
            self._provenance()
            with self.gateway:
                for evaluator in self.state.task.evaluators:
                    if evaluator.tool not in self.gateway.catalog:
                        raise ValueError(f"Required evaluator tool unavailable: {evaluator.tool}")
                for candidate in self.state.task.candidates:
                    self.submit(candidate.model_dump())
                if self.state.mode == "biomni":
                    if planner_factory is None:
                        from biosyn.agent.planner import BiomniPlanner
                        planner_factory = BiomniPlanner
                    planner = planner_factory(self.config, self)
                    for index in range(1, self.state.task.max_rounds + 1):
                        self.state.round = index
                        self.store.snapshot(self.state)
                        answer = planner.run_round(index)
                        self.store.write(f"answer-{index}.json", {"answer": answer})
                        if self.state.best_id is not None or self.gateway.calls >= self.state.task.max_tool_calls:
                            break
                        self.store.event("reflection_required", round=index, feedback=self.summary())
                self.state.status = "passed" if self.state.best_id is not None else "exhausted"
        except Exception as exc:
            self.state.status = "failed"
            self.state.error = f"{type(exc).__name__}: {exc}"
            self.store.event("run_failed", error=self.state.error)
            raise
        finally:
            self.store.event("run_finished", status=self.state.status, error=self.state.error)
            self.store.snapshot(self.state)
            self.store.write("report.json", self.state.model_dump(mode="json"))
            if self.state.best_id:
                best = next(c for c in self.state.candidates if c.id == self.state.best_id)
                self.store.write("best-design.json", best.model_dump())
                fasta = "".join(f">{best.id}|{kind}\n{getattr(best, kind)}\n" for kind in
                                ("dna", "rna", "protein") if getattr(best, kind))
                (self.store.path / "best-design.fasta").write_text(fasta)
        return self.state

    def _provenance(self):
        import biomni
        package_root = Path(biomni.__file__).resolve().parent.parent
        result = subprocess.run(["git", "-C", str(package_root), "rev-parse", "HEAD"],
                                capture_output=True, text=True)
        self.store.write("provenance.json", {"python": platform.python_version(),
            "biomni_path": str(package_root),
            "biomni_commit": result.stdout.strip() if result.returncode == 0 else None,
            "packages": {p: version(p) for p in ("biosyn-agent", "biomni", "mcp", "pydantic", "langgraph", "langchain-openai")}})
