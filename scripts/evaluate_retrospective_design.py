#!/usr/bin/env python3
"""Compare design structure, sequences and supplied historical references."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from Bio.Seq import Seq

OUT = Path(__file__).resolve().parents[1] / "tasks/application_suite_v3/retrospective"


def protein(cds):
    cds = cds.upper().replace("U", "T")
    if not cds or set(cds) - set("ACGT") or len(cds) % 3:
        raise ValueError("Invalid complete CDS")
    aa = str(Seq(cds).translate())
    if not aa.startswith("M") or not aa.endswith("*") or "*" in aa[:-1]:
        raise ValueError("CDS must start with methionine and have one terminal stop")
    return aa[:-1]

def simulate(nodes, inputs):
    """Evaluate the submitted graph, independent of stored node truth strings."""
    by_id = {n["id"]: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("Duplicate node IDs")
    memo, visiting = {}, set()

    def visit(nid):
        if nid in memo:
            return memo[nid]
        if nid in visiting:
            raise ValueError("Combinational circuit has a cycle")
        visiting.add(nid)
        node = by_id[nid]
        kind, deps = node["type"], node["inputs"]
        if kind == "INPUT":
            if deps or node["name"] not in inputs:
                raise ValueError("Invalid input node")
            result = inputs[node["name"]]
            if type(result) is not int or result not in (0, 1):
                raise ValueError("Input is not Boolean")
        else:
            vals = [visit(x) for x in deps]
            if kind == "NOT" and len(vals) == 1:
                result = 1 - vals[0]
            elif kind == "NOR" and len(vals) >= 1:
                result = int(not any(vals))
            elif kind == "OUTPUT" and len(vals) == 1:
                result = vals[0]
            elif kind == "OUTPUT_OR" and len(vals) >= 1:
                result = int(any(vals))
            elif kind in {"AND", "OR", "XOR", "NAND", "XNOR"} and len(vals) >= 1:
                if kind in {"AND", "NAND"}:
                    result = int(all(vals))
                elif kind == "OR":
                    result = int(any(vals))
                else:
                    result = sum(vals) % 2
                if kind in {"NAND", "XNOR"}:
                    result = 1 - result
            else:
                raise ValueError(f"Unsupported gate or arity: {kind}")
        visiting.remove(nid)
        memo[nid] = result
        return result

    for node in nodes:
        visit(node["id"])
    return {n["name"]: memo[n["id"]] for n in nodes if n["type"].startswith("OUTPUT")}


def f1(a, b):
    a, b = set(a), set(b)
    return 2 * len(a & b) / (len(a) + len(b)) if a or b else 1.0


def dna(value):
    if not isinstance(value, str):
        raise ValueError("Sequence must be a string")
    value = value.upper().replace("U", "T")
    if not value or set(value) - set("ACGT"):
        raise ValueError("Sequence contains invalid nucleotides")
    return value


def reference_identity(candidate, reference, fields):
    for key in fields:
        if key not in candidate or key not in reference:
            return False
        if key in {"cds", "rna_body", "assay_barcode", "dna_sequence"}:
            if dna(candidate[key]) != dna(reference[key]):
                return False
        elif candidate[key] != reference[key]:
            return False
    return True


def compare(task, private, submission):
    """Invalid designs raise ValueError; biological predictions remain explicitly unavailable."""
    track, inputs = task["track"], task["inputs"]
    fields = task["expected_output"]["required_fields"]
    missing = set(fields) - set(submission)
    if missing:
        raise ValueError(f"Missing submission fields: {sorted(missing)}")
    result = {"case_id": task["case_id"], "structural_checks": {}, "reference_similarity": [],
              "matched_historical_references": [], "candidate_experimental_outcome": "not_measured",
              "candidate_phenotype_prediction": "not_computed",
              "interpretation": "Similarity and formal checks do not imply biological efficacy. Matched observations are historical records in the documented context."}
    checks = result["structural_checks"]
    if track in {"bacterial_cds", "mammalian_mrna"}:
        cds = dna(submission["cds"])
        checks["protein_preserved"] = protein(cds) == inputs["protein_sequence"]
        if not checks["protein_preserved"]:
            raise ValueError("Encoded protein differs from the requested full protein")
        if track == "bacterial_cds":
            checks["terminal_stop_preserved"] = cds.endswith(inputs["required_terminal_stop"])
            if not checks["terminal_stop_preserved"]:
                raise ValueError("Required fixed-context stop changed")
            checks["fixed_context"] = "Supplied upstream/downstream contexts retained by task contract; output is CDS only"
        else:
            barcode = dna(submission["assay_barcode"])
            if len(barcode) != inputs["assay_barcode_length"]:
                raise ValueError("Experimental barcode has wrong length")
            expected = dna(inputs["fixed_rna_prefix"]) + cds + dna(inputs["fixed_suffix_before_barcode"]) + barcode + dna(inputs["fixed_suffix_after_barcode"])
            checks["rna_assembly_preserved"] = dna(submission["rna_body"]) == expected
            if not checks["rna_assembly_preserved"]:
                raise ValueError("RNA body does not match fixed scaffold and submitted CDS/barcode")
        for ref in private["references"]:
            seq = dna(ref["design"]["cds"])
            identity = sum(x == y for x, y in zip(seq, cds)) / len(cds) if len(seq) == len(cds) else None
            result["reference_similarity"].append({"reference_id": ref["reference_id"],
                "CDS_nucleotide_identity": identity, "scope": "CDS only; fixed flanks excluded",
                "descriptive_only": True})
    elif track == "genetic_circuit":
        dna(submission["dna_sequence"])
        nodes = submission["nodes"]
        outputs = {n["name"] for n in nodes if n["type"].startswith("OUTPUT")}
        if outputs != set(inputs["reporters"]):
            raise ValueError("Wrong reporter output set")
        if len(outputs) != sum(n["type"].startswith("OUTPUT") for n in nodes):
            raise ValueError("Duplicate reporter names")
        observed_inputs = [n["name"] for n in nodes if n["type"] == "INPUT"]
        if sorted(observed_inputs) != sorted(inputs["input_signals"]):
            raise ValueError("Wrong or duplicate physical sensor inputs")
        truth = inputs["truth_table"]
        correct = sum(simulate(nodes, row["inputs"]) == row["outputs"] for row in truth)
        checks.update(boolean_state_accuracy=correct / len(truth), dna_alphabet_valid=True,
                      dna_graph_correspondence="not_evaluated", analog_expression="not_evaluated")
        for ref in private["references"]:
            result["reference_similarity"].append({"reference_id": ref["reference_id"],
                "DNA_insert_exact_match": dna(ref["design"]["dna_sequence"]) == dna(submission["dna_sequence"])})
    elif track == "metabolic_pathway":
        steps = submission["steps"]
        if not isinstance(steps, list) or not steps:
            raise ValueError("Pathway must contain directed steps")
        edges, genes = [], set()
        for step in steps:
            names = step.get("genes", [step["gene"]] if "gene" in step else [])
            if not names or not step.get("source") or not step.get("input") or not step.get("output"):
                raise ValueError("Each step needs substrates, products, genes and organism source")
            edges.append((step["input"], step["output"]))
            genes.update((name, step["source"]) for name in names)
        checks["introduced_subpath_connected"] = all(a[1] == b[0] for a, b in zip(edges, edges[1:]))
        checks["reaches_requested_product"] = edges[-1][1] == inputs["product"]
        checks["pathway_length"] = len(steps)
        checks["heterologous_gene_count_proxy"] = sum(not source.startswith("Escherichia coli") for _, source in genes)
        checks["glucose_to_first_intermediate"] = "requires host-model verification"
        checks["stoichiometric_yield"] = "not_computed"
        for ref in private["references"]:
            rsteps = ref["design"]["steps"]
            redges = [(s["input"], s["output"]) for s in rsteps]
            rgenes = {(g, s["source"]) for s in rsteps for g in s.get("genes", [s["gene"]] if "gene" in s else [])}
            result["reference_similarity"].append({"reference_id": ref["reference_id"],
                "directed_main_reaction_edge_F1": f1(edges, redges), "gene_source_set_F1": f1(genes, rgenes),
                "scope": "Named main transformations; no inferred full stoichiometry or gene-name synonym equivalence"})
    else:
        raise ValueError(f"Unknown task track: {track}")
    if track == "metabolic_pathway":
        result["historical_lookup_status"] = "disabled: pathway plan is not a fully specified strain/construct"
    for ref in private["references"]:
        if track != "metabolic_pathway" and reference_identity(submission, ref["design"], fields):
            result["matched_historical_references"].append({"reference_id": ref["reference_id"],
                "observations": ref["observations"], "applicability": "Historical source context only; not a prospective measurement of this submission"})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=OUT)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if Path(args.case_id).name != args.case_id:
        parser.error("case-id must be a simple case identifier")
    task = json.loads((args.dataset / "public" / f"{args.case_id}.json").read_text())
    private = json.loads((args.dataset / "private" / f"{args.case_id}.json").read_text())
    report = compare(task, private, json.loads(args.submission.read_text()))
    payload = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(payload)
    else:
        print(payload)


if __name__ == "__main__":
    main()
