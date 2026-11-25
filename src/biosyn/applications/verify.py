"""Fresh-process computational replay; explicitly not independent biological validation."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from biosyn.applications.science import Evaluator


def verify(case, profile, resources):
    ledger_path = case / "ledger/ledger.json"
    if not ledger_path.exists():
        raise ValueError("No evaluated candidate ledger")
    ledger = json.loads(ledger_path.read_text())
    evaluator = Evaluator(profile, resources)
    checks = []
    valid = {}
    for row in ledger["history"]:
        candidate = row["candidate"]
        if not row["evaluation"].get("valid"):
            continue
        if hashlib.sha256(json.dumps(candidate["design"], sort_keys=True).encode()).hexdigest() != row["design_sha256"]:
            raise ValueError("Immutable design hash changed")
        new = evaluator.evaluate(candidate["design"])
        original = row["evaluation"]
        agreement = (new["valid"] == original["valid"] and new["feasible"] == original["feasible"] and
                     set(new["metrics"]) == set(original["metrics"]) and
                     all(math.isclose(float(new["metrics"][k]), float(v), abs_tol=1e-5, rel_tol=1e-5)
                         for k, v in original["metrics"].items()))
        checks.append({"candidate_id": candidate["id"], "matches": agreement, "recomputed": new})
        if not agreement:
            raise ValueError(f"Replay disagreement for {candidate['id']}")
        valid[candidate["id"]] = candidate
    revised = [c["id"] for c in valid.values() if c["parent_id"] in valid and
               c["design"] != valid[c["parent_id"]]["design"] and c["rationale"].strip()]
    final = ledger["final"]
    sensitivity = []
    if final:
        design = valid[final["candidate_id"]]["design"]
        if evaluator.profile["kind"] in ("circuit", "metabolic"):
            for scale in (0.8, 1.2):
                method = evaluator.circuit if evaluator.profile["kind"] == "circuit" else evaluator.metabolic
                sensitivity.append({"scale": scale, "result": method(design, scale)})
    result = {"fresh_process_replay": True, "all_metrics_reproduced": True,
              "evaluated_design_count": len(valid), "changed_parent_linked_designs": revised,
              "finalized": final is not None,
              "computational_loop_structurally_complete": bool(revised and final),
              "feedback_influenced_decision": "Requires inspection of recorded Biomni messages and rationales; structure alone does not prove causality.",
              "independent_biological_validation": False, "checks": checks,
              "declared_sensitivity_scenarios": sensitivity}
    (case / "verification.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("checks", "declared_sensitivity_scenarios")}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=Path, required=True)
    parser.add_argument("--profile", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    args = parser.parse_args()
    verify(args.case, args.profile, args.resources)


if __name__ == "__main__":
    main()
