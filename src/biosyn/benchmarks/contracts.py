"""Public submission checks only. No private outcomes, reference designs or scores."""
import json
import re

from Bio.Seq import Seq


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_edits(edits, regions, sequence):
    require(isinstance(edits, list), "edits must be a list")
    by_name = {r["name"]: r for r in regions}
    seen = set()
    chars = list(sequence)
    for edit in edits:
        require(isinstance(edit, dict) and set(edit) == {"region", "codon_index", "codon"},
                "Each edit requires region, codon_index and codon only")
        region = by_name.get(edit["region"])
        require(region is not None, "Unknown CDS region")
        index = edit["codon_index"]
        require(type(index) is int and 0 <= index < (region["end"] - region["start"]) // 3,
                "codon_index outside CDS")
        require((edit["region"], index) not in seen, "Duplicate codon edit")
        seen.add((edit["region"], index))
        codon = edit["codon"]
        require(isinstance(codon, str) and re.fullmatch("[ACGT]{3}", codon), "Invalid codon")
        start = region["start"] + 3 * index
        require(str(Seq(sequence[start:start + 3]).translate(table=11)) ==
                str(Seq(codon).translate(table=11)), "Edit is not synonymous")
        chars[start:start + 3] = codon
    return "".join(chars)


def validate_json(task, submission, catalog=None):
    kind = task["task_type"]
    if kind == "T5":
        require(isinstance(submission, list) and submission, "Expected nonempty truth-table list")
        for row in submission:
            require(isinstance(row, dict) and set(row) == {"inputs", "outputs"}, "Invalid table row")
            require(all(isinstance(row[k], dict) and all(type(v) is int and v in (0, 1)
                    for v in row[k].values()) for k in ("inputs", "outputs")), "Invalid truth values")
        require(len({json.dumps(r["inputs"], sort_keys=True) for r in submission}) == len(submission),
                "Duplicate truth-table inputs")
    elif kind == "E1":
        require(isinstance(submission, dict), "Submission must be a JSON object")
        pid = submission.get("promoter_id")
        require(pid in task["candidate_promoter_ids"], "Promoter outside allowed candidates")
        parts = {p["id"]: p for p in catalog["parts"]}
        bg = next(b for b in catalog["backgrounds"] if b["id"] == task["background_id"])
        chunks = [("ascI_site", "GGCGCGCC"), ("promoter_region", parts[pid]["sequence"]),
                  ("rbs_region", bg["rbs_sequence"]), ("ndeI_prefix", "CAT"),
                  ("sfGFP_CDS", bg["cds_sequence"])]
        sequence, features = "", []
        for name, chunk in chunks:
            features.append({"name": name, "start": len(sequence), "end": len(sequence) + len(chunk), "strand": 1})
            sequence += chunk
        expected = {"background_id": bg["id"], "promoter_id": pid,
                    "artifact_type": "regulatory_CDS_fragment", "sequence": sequence, "features": features}
        require(all(submission.get(k) == v for k, v in expected.items()), "E1 fragment/context/coordinates mismatch")
    elif kind == "expression_construct_v2":
        require(isinstance(submission, dict) and set(submission) == {"edits", "rationale"},
                "Expected edits and rationale")
        require(isinstance(submission["rationale"], str), "rationale must be a string")
        spec = task["spec"]
        check_edits(submission["edits"], spec["cds_regions"], spec["template_sequence"])
        require(len(submission["edits"]) <= spec["max_codon_edits"], "Codon edit budget exceeded")
    elif kind == "expression_diagnosis":
        require(isinstance(submission, dict) and set(submission) == {"limiting_level", "evidence", "repair"},
                "Expected limiting_level, evidence and repair")
        level = submission["limiting_level"]
        require(level in ("dna", "rna", "protein") and isinstance(submission["evidence"], list),
                "Invalid declared level or evidence")
        repair = submission["repair"]
        require(isinstance(repair, dict) and repair.get("level") == level, "Repair must target declared level")
        if "replacement_part_id" in repair:
            require(set(repair) == {"level", "replacement_part_id"}, "Unexpected repair fields")
            require(repair["replacement_part_id"] in {p["id"] for p in task["spec"]["catalogues"][level]},
                    "Unknown catalogue part")
        elif "replacement_sequence" in repair:
            require(set(repair) == {"level", "replacement_sequence"} and
                    re.fullmatch("[ACGT]+", repair["replacement_sequence"]), "Invalid replacement sequence")
        else:
            raise ValueError("Repair requires replacement_part_id or replacement_sequence")
    elif kind == "metabolic_pathway_v2":
        require(isinstance(submission, dict) and set(submission) ==
                {"enzymes", "acknowledged_assumptions", "rationale"}, "Invalid pathway submission fields")
        spec = task["spec"]
        require(isinstance(submission["enzymes"], list) and len(submission["enzymes"]) == len(spec["steps"]),
                "One enzyme per step required")
        for item, step in zip(submission["enzymes"], spec["steps"]):
            require(set(item) == {"reaction_id", "accession", "evidence", "edits"} and
                    item["reaction_id"] == step["reaction_id"], "Pathway step/field mismatch")
            entry = next((x for x in step["enzyme_candidates"] if x["accession"] == item["accession"]), None)
            require(entry is not None, "Enzyme outside public catalogue")
            require(isinstance(item["evidence"], dict) and set(item["evidence"]) ==
                    {"rhea", "evidence_code", "source"}, "Invalid evidence fields")
            require(all(isinstance(v, str) for v in item["evidence"].values()), "Evidence fields must be strings")
            check_edits(item["edits"], [{"name": item["accession"], "start": 0, "end": len(entry["cds"])}], entry["cds"])
        require(isinstance(submission["acknowledged_assumptions"], list) and
                all(isinstance(x, str) for x in submission["acknowledged_assumptions"]),
                "assumptions must be a string list")
        require(isinstance(submission["rationale"], str), "rationale must be a string")
    else:
        raise ValueError(f"Unsupported JSON contract: {kind}")
    return {"contract_valid": True, "task_success": None, "scope": "public submission checks only"}


def instructions(kind, answer_format):
    if answer_format == "python":
        return ("Write a complete self-contained Python program using sbol3. It must write its SBOL document "
                "to os.environ['BIOSYN_OUTPUT'] in sbol3.SORTED_NTRIPLES format. Submit with "
                "biosyn_submit(candidate={'program_path':'/work/submission.py'}). The submit tool executes "
                "the saved program and checks SBOL syntax, without returning the private task grade.")
    formats = {
        "T5": "A JSON list of complete truth-table rows {'inputs':{...},'outputs':{...}}.",
        "E1": "The JSON fragment with background_id, promoter_id, artifact_type, sequence, features specified in protocol.md.",
        "expression_construct_v2": "JSON {'edits':[{'region':CDS_name,'codon_index':zero_based_index,'codon':'ACG'}], 'rationale':str}. Use only synonymous changes within max_codon_edits; pursue all public objective bands.",
        "expression_diagnosis": "JSON {'limiting_level':'dna'|'rna'|'protein','evidence':list,'repair':{'level':same_level,'replacement_part_id':catalogue_id}}. For a de novo repair use replacement_sequence instead of replacement_part_id.",
        "metabolic_pathway_v2": "JSON {'enzymes':[{'reaction_id':str,'accession':str,'evidence':{'rhea':str,'evidence_code':str,'source':str},'edits':[{'region':accession,'codon_index':int,'codon':str}]}], 'acknowledged_assumptions':[str], 'rationale':str}. Enzymes follow task step order.",
    }
    return (formats[kind] + " Write the JSON to /work/submission.json, then finalize with "
            "biosyn_submit(candidate={'path':'/work/submission.json'}). Public validation is not task success.")
