"""Final-artifact rescoring in a new MCP process, outside every method's feedback budget."""

import math

from biosyn.agent.state import RunStore
from biosyn.mcp.client import ToolGateway, ToolError


def compare_values(left, right, path="root", atol=1e-6, rtol=1e-6):
    differences = []
    if isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            differences.append(path + ": keys differ")
        for key in left.keys() & right.keys():
            differences.extend(compare_values(left[key], right[key], path + "." + key, atol, rtol))
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            differences.append(path + ": lengths differ")
        for i, (a, b) in enumerate(zip(left, right)):
            differences.extend(compare_values(a, b, f"{path}[{i}]", atol, rtol))
    elif (
        isinstance(left, (float, int))
        and not isinstance(left, bool)
        and isinstance(right, (float, int))
        and not isinstance(right, bool)
    ):
        if not math.isclose(left, right, rel_tol=rtol, abs_tol=atol):
            differences.append(f"{path}: {left} != {right}")
    elif left != right:
        differences.append(path + ": values differ")
    return differences


def rescore_final(settings, case, artifact, original):
    store = RunStore(case / "independent-rescore")
    with ToolGateway({"science": settings}, store, max_calls=1) as gateway:
        try:
            scored = gateway.call("science.evaluate_design", {"design": artifact})["data"]
        except (ToolError, ValueError, TimeoutError) as exc:
            result = {"completed": False, "consistent": False, "error": f"{type(exc).__name__}: {exc}"[:3000]}
        else:
            differences = compare_values(original, scored)
            result = {
                "completed": True,
                "consistent": not differences,
                "differences": differences,
                "tolerance": {"absolute": 1e-6, "relative": 1e-6},
                "evaluation": scored,
                "scope": "Fresh process and same scientific model; not independent biological evidence.",
            }
    store.write("result.json", result)
    return result
