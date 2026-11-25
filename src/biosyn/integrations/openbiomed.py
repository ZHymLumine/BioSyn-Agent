"""Expose explicitly configured OpenBioMed workflows through MCP.

Run in the OpenBioMed environment, with that checkout installed. Workflows are
selected by name from BIOSYN_OPENBIOMED_WORKFLOWS, never supplied as executable code.
"""
import contextlib
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from biosyn.agent.state import StrictModel
from biosyn.mcp.common import serve

server = FastMCP("BioSyn OpenBioMed workflows", log_level="WARNING")
_lock = threading.Lock()


class WorkflowInput(StrictModel):
    node: int = Field(ge=0)
    field: str
    kind: Literal["json", "protein", "molecule", "text"] = "json"
    value: Any


def workflow_paths():
    path = os.environ.get("BIOSYN_OPENBIOMED_WORKFLOWS")
    if not path:
        raise ValueError("BIOSYN_OPENBIOMED_WORKFLOWS must point to a name-to-YAML JSON mapping")
    mapping = json.loads(Path(path).read_text())
    if not isinstance(mapping, dict) or not all(isinstance(v, str) for v in mapping.values()):
        raise ValueError("Workflow mapping must contain string paths")
    return mapping


def encode_output(value):
    from open_biomed.data import Molecule, Protein, Text
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, (list, tuple)):
        return [encode_output(v) for v in value]
    if isinstance(value, dict):
        return {str(k): encode_output(v) for k, v in value.items()}
    if isinstance(value, Protein):
        if value.all_atom is not None:
            raise TypeError("Structural Protein output requires an explicit artifact export node in the workflow")
        return {"type": "protein", "sequence": value.sequence}
    if isinstance(value, Molecule):
        return {"type": "molecule", "smiles": value.smiles}
    if isinstance(value, Text):
        return {"type": "text", "text": str(value)}
    raise TypeError(f"Workflow returned unsupported output type {type(value).__name__}; add an export node")


@server.tool()
def list_workflows() -> dict:
    """List locally configured OpenBioMed workflows and whether their YAML files exist."""
    return {name: {"path": path, "exists": Path(path).is_file()} for name, path in workflow_paths().items()}


@server.tool()
def run_workflow(name: str, inputs: list[WorkflowInput], repeats: int = 1) -> dict:
    """Execute a named OpenBioMed Workflow using typed inputs; failures remain MCP errors."""
    paths = workflow_paths()
    if name not in paths:
        raise ValueError(f"Workflow {name!r} is not configured")
    if not 1 <= repeats <= 10:
        raise ValueError("repeats must be between 1 and 10")
    path = Path(paths[name]).resolve(strict=True)
    with _lock, contextlib.redirect_stdout(sys.stderr):
        from open_biomed.core.workflow import Workflow
        from open_biomed.utils.config import Config
        from open_biomed.data import Molecule, Protein, Text
        values = []
        for entry in inputs:
            value = entry.value
            if entry.kind == "protein":
                value = Protein.from_fasta(value)
            elif entry.kind == "molecule":
                value = Molecule.from_smiles(value)
            elif entry.kind == "text":
                value = Text.from_str(value)
            values.append((entry.node, entry.field, value))
        workflow = Workflow(Config(config_file=str(path)))
        for node, field, _ in values:
            if node >= len(workflow.nodes):
                raise ValueError(f"Input node {node} is outside workflow")
        outputs, messages = workflow.run(values, num_repeats=repeats)
        result = encode_output(outputs)
    return {"workflow": name, "workflow_file": str(path), "outputs": result,
            "messages": messages, "evidence_type": "workflow_output"}


if __name__ == "__main__":
    workflow_paths()
    serve(server)
