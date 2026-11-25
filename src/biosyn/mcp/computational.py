"""Task-bound scientific MCP tools for the three-method comparison."""

import argparse
import hashlib
import json
from pathlib import Path

from mcp.server.fastmcp import FastMCP
from biosyn.applications.computational_circuit import CircuitEvaluator, synthesize_nor, validate_truth_table
from biosyn.applications.computational_sequence import SequenceEvaluator, check_cds
from biosyn.applications.computational_resources import verify_resources


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--resource-lock", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    task_bytes = args.task.read_bytes()
    task = json.loads(task_bytes)
    task_sha = hashlib.sha256(task_bytes).hexdigest()
    resource_receipt = verify_resources(args.root, args.resource_lock, task["track"])
    regulatory = task.get("evaluation_engine") in {"emopec_sd", "optimus_utr"}
    if regulatory:
        from biosyn.applications.computational_regulatory import RegulatoryEvaluator
        evaluator = RegulatoryEvaluator(task, args.catalog, args.resources)
    elif task["track"] == "genetic_circuit":
        validate_truth_table(task["inputs"]["truth_table"], task["inputs"]["input_signals"])
        evaluator = CircuitEvaluator(args.catalog)
    elif task["track"] in ("bacterial_cds", "mammalian_mrna"):
        evaluator = SequenceEvaluator(args.resources)
    else:
        from biosyn.applications.computational_metabolic import MetabolicEvaluator

        evaluator = MetabolicEvaluator(task, args.catalog, args.resources)
    server = FastMCP("BioSyn computational evaluation")

    def materialize(design):
        if regulatory:
            return evaluator.materialize(task, design)
        if task["track"] == "genetic_circuit":
            artifact = evaluator.assemble(
                design["gates"], design["output"], list(task["inputs"]["input_signals"])
            )
        elif task["track"] == "mammalian_mrna":
            i = task["inputs"]
            cds = check_cds(task, design)
            barcode = design.get("assay_barcode", i.get("fixed_assay_barcode", "")).replace("T", "U")
            if len(barcode) != i["assay_barcode_length"] or set(barcode) - set("ACGU"):
                raise ValueError("Invalid assay barcode")
            if i.get("fixed_assay_barcode") and barcode != i["fixed_assay_barcode"].replace("T", "U"):
                raise ValueError("The task fixes the assay barcode")
            artifact = dict(
                design,
                assay_barcode=barcode,
                rna_body=i["fixed_rna_prefix"]
                + cds.replace("T", "U")
                + i["fixed_suffix_before_barcode"]
                + barcode
                + i["fixed_suffix_after_barcode"],
            )
        else:
            artifact = dict(design)
        for field in ("dna_sequence", "rna_body", "assembly", "nodes"):
            if field in design and artifact.get(field) != design[field]:
                raise ValueError(
                    "Supplied " + field + " conflicts with deterministic assembly; it was not repaired"
                )
        return artifact

    @server.tool()
    def evaluate_design(design: dict) -> dict:
        """Compute declared scientific metrics for one complete design; model predictions are not measurements."""
        artifact = materialize(design)
        result = evaluator.evaluate(task, artifact)
        from biosyn.applications.goal_contract import assess_goals
        result.update(assess_goals(task, result))
        result["artifact"] = artifact
        result["public_task_sha256"] = task_sha
        result["resource_verification"] = resource_receipt
        json.dumps(result, allow_nan=False)
        return result

    if task["track"] == "genetic_circuit":

        @server.tool()
        def synthesize_circuit() -> dict:
            """Synthesize a generic NOR topology from the public truth table. Choose real gate variants yourself."""
            return synthesize_nor(task["inputs"]["truth_table"])

        @server.tool()
        def assemble_circuit(gates: list[dict], output: str) -> dict:
            """Expand selected physical NOR gates into original Cello devices, DNA and a consistent logic graph."""
            return evaluator.assemble(gates, output, list(task["inputs"]["input_signals"]))

    if regulatory:

        @server.tool()
        def assemble_regulatory_design(design: dict) -> dict:
            """Assemble the fixed source context around the selected regulatory region, without scoring."""
            return materialize(design)

    if task["track"] == "mammalian_mrna" and not regulatory:

        @server.tool()
        def assemble_mrna(cds: str, assay_barcode: str | None = None) -> dict:
            """Assemble the fixed RNA scaffold and declared barcode; no phenotype prediction."""
            design = {"cds": cds}
            if assay_barcode is not None:
                design["assay_barcode"] = assay_barcode
            return materialize(design)

    if task["track"] == "metabolic_pathway":

        @server.tool()
        def enumerate_pathways() -> dict:
            """Enumerate connected routes in the public reaction library; performs no FBA scoring."""
            from biosyn.applications.computational_metabolic import routes_to_product

            return {"routes": routes_to_product(task, evaluator.catalog)}

    server.run()


if __name__ == "__main__":
    main()
