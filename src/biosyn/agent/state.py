from __future__ import annotations

import json
import os
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Candidate(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    parent_id: str | None = None
    dna: str | None = None
    rna: str | None = None
    protein: str | None = None
    phenotype: dict[str, Any] = Field(default_factory=dict)
    artifacts: dict[str, str] = Field(default_factory=dict)
    rationale: str = ""

    @model_validator(mode="after")
    def validate_sequences(self):
        if not any((self.dna, self.rna, self.protein, self.artifacts)):
            raise ValueError("A candidate needs a sequence or an artifact")
        for name, alphabet in (("dna", "ACGT"), ("rna", "ACGU"),
                               ("protein", "ACDEFGHIKLMNPQRSTVWY")):
            value = getattr(self, name)
            if value is not None and (not value or not re.fullmatch(f"[{alphabet}]+", value)):
                raise ValueError(f"{name} must be a nonempty uppercase canonical sequence")
        return self


class Objective(StrictModel):
    metric: str
    direction: Literal["maximize", "minimize"]
    lower: FiniteFloat
    upper: FiniteFloat
    weight: FiniteFloat = Field(default=1, gt=0)
    minimum: FiniteFloat | None = None
    maximum: FiniteFloat | None = None

    @model_validator(mode="after")
    def bounds(self):
        if self.lower >= self.upper:
            raise ValueError("Normalization requires lower < upper")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum must be <= maximum")
        return self


class EvaluationSpec(StrictModel):
    name: str
    tool: str = Field(description="MCP server.tool")
    inputs: dict[str, str] = Field(description="Tool argument -> candidate field, e.g. sequence: rna")
    arguments: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, str] = Field(description="Metric name -> dotted path in MCP structured result")

    @model_validator(mode="after")
    def disjoint_arguments(self):
        if self.inputs.keys() & self.arguments.keys():
            raise ValueError("Candidate inputs and constant arguments overlap")
        return self


class TaskSpec(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
    request: str = Field(min_length=1)
    host: str | None = None
    objectives: list[Objective] = Field(min_length=1)
    evaluators: list[EvaluationSpec] = Field(min_length=1)
    candidates: list[Candidate] = Field(default_factory=list)
    require_transcription_match: bool = False
    require_translation_match: bool = False
    translation_table: int = Field(default=1, ge=1)
    required_protein: str | None = Field(default=None, pattern=r"^[ACDEFGHIKLMNPQRSTVWY]+$")
    max_rounds: int = Field(default=3, ge=1, le=20)
    max_tool_calls: int = Field(default=40, ge=1, le=10000)
    max_candidates: int = Field(default=20, ge=1, le=1000)

    @model_validator(mode="after")
    def unique_names(self):
        for values in ([c.id for c in self.candidates], [e.name for e in self.evaluators],
                       [o.metric for o in self.objectives]):
            if len(values) != len(set(values)):
                raise ValueError("Duplicate candidate, evaluator, or objective name")
        metrics = [m for e in self.evaluators for m in e.metrics]
        if len(metrics) != len(set(metrics)):
            raise ValueError("Each metric must have exactly one evaluator")
        if not {o.metric for o in self.objectives} <= set(metrics):
            raise ValueError("Every objective must have an evaluator")
        if len(self.candidates) > self.max_candidates:
            raise ValueError("Initial candidates exceed budget")
        return self


class Evaluation(StrictModel):
    candidate_id: str
    metrics: dict[str, FiniteFloat] = Field(default_factory=dict)
    call_ids: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    score: FiniteFloat | None = None
    passed: bool = False


class DesignState(StrictModel):
    schema_version: str = "1"
    run_id: str
    task: TaskSpec
    mode: Literal["biomni", "evaluate"]
    status: Literal["running", "passed", "exhausted", "failed"] = "running"
    round: int = 0
    candidates: list[Candidate] = Field(default_factory=list)
    evaluations: dict[str, Evaluation] = Field(default_factory=dict)
    best_id: str | None = None
    pareto_ids: list[str] = Field(default_factory=list)
    error: str | None = None


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    """Exclusive run directory, atomic snapshots, append-only tool/event records."""

    def __init__(self, path: Path):
        self.path = path.resolve()
        self.path.mkdir(parents=True, exist_ok=False)
        self._lock = threading.RLock()

    def write(self, name: str, value: Any):
        with self._lock:
            destination = self.path / name
            temp = destination.with_suffix(destination.suffix + ".tmp")
            with temp.open("w") as handle:
                json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            temp.replace(destination)

    def event(self, kind: str, **payload):
        with self._lock, (self.path / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"time": timestamp(), "kind": kind, **payload},
                                    ensure_ascii=False, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def snapshot(self, state: DesignState):
        self.write("state.json", state.model_dump(mode="json"))
