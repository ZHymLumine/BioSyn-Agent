"""Evaluate public numerical goals without conflating prediction and measurement."""

import math


def assess_goals(task, result):
    contract = task.get("evaluation_contract")
    if contract is None:
        return {"goal_met": None, "goal_evaluation": "No numerical goal contract was registered"}
    if not result.get("valid", False):
        return {"goal_met": False, "goal_evaluation": "Invalid design"}
    if not contract.get("requirements") or not contract.get("evidence_type"):
        raise ValueError("A goal contract requires evidence_type and explicit requirements")
    checks = []
    for requirement in contract["requirements"]:
        metric = requirement["metric"]
        expected = requirement["operator"]
        target = requirement["value"]
        value = result.get("metrics", {}).get(metric)
        tolerance = requirement.get("absolute_tolerance", 0.0)
        if not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Goal tolerance must be finite and nonnegative")
        if expected not in {">=", "<=", ">", "<", "=="}:
            raise ValueError("Unsupported goal operator")
        if not isinstance(target, (int, float)) or not math.isfinite(target):
            raise ValueError("Goal target must be a finite number")
        if value is None or not isinstance(value, (int, float)) or not math.isfinite(value):
            passed = None
        elif expected == ">=":
            passed = value >= target - tolerance
        elif expected == "<=":
            passed = value <= target + tolerance
        elif expected == ">":
            passed = value > target + tolerance
        elif expected == "<":
            passed = value < target - tolerance
        else:
            passed = abs(value - target) <= tolerance
        checks.append({"metric": metric, "value": value, "operator": expected, "target": target,
                       "absolute_tolerance": tolerance, "met": passed})
    met = False if any(c["met"] is False for c in checks) else (
        None if any(c["met"] is None for c in checks) else True)
    if result.get("feasible") is False:
        met = False
    return {"goal_met": met, "goal_requirements": checks,
            "goal_evidence_type": contract["evidence_type"],
            "goal_evaluation": "public_requirements", "experimental_outcome": "not_measured"}
