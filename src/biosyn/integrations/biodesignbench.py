"""Export real execution evidence in the public AgentOutput shape, without inventing benchmark scores."""
import json
from datetime import datetime
from pathlib import Path


def export_output(run_dir: Path, output: Path):
    report = json.loads((run_dir / "report.json").read_text())
    events = [json.loads(line) for line in (run_dir / "events.jsonl").read_text().splitlines()]
    started = {e["call_id"]: e for e in events if e["kind"] == "tool_started"}
    calls = []
    iteration = 1
    for event in events:
        if event["kind"] == "reflection_required":
            iteration = event["round"] + 1
        if event["kind"] not in {"tool_succeeded", "tool_failed"}:
            continue
        args = started[event["call_id"]]["arguments"]
        values = {k: v for k, v in args.items() if isinstance(v, (bool, int, float)) or
                  (isinstance(v, str) and len(v) <= 60 and "\n" not in v)}
        calls.append({"tool": event["tool"].split(".", 1)[1], "iteration": iteration,
                      "success": event["kind"] == "tool_succeeded", "error": event.get("error"),
                      "args_summary": {k: type(v).__name__ for k, v in args.items()},
                      "args_values": values, "duration_seconds": event["elapsed_seconds"]})
    best = next((c for c in report["candidates"] if c["id"] == report["best_id"]), None)
    designs = []
    if best:
        designs = [{"id": best["id"], "sequence": best["protein"], "dna": best["dna"],
                    "rna": best["rna"], "artifacts": best["artifacts"],
                    "metrics": report["evaluations"][best["id"]]["metrics"]}]
    llm_calls = [e for e in events if e["kind"] == "llm_completed"]
    wall_time = (datetime.fromisoformat(events[-1]["time"]) - datetime.fromisoformat(events[0]["time"])).total_seconds() if events else 0
    result = {"code": "", "designs": designs, "artifacts": list(best["artifacts"].values()) if best else [],
              "output_dir": str(run_dir.resolve()), "tools_used": sorted({c["tool"] for c in calls}),
              "tool_call_log": calls, "api_calls": len(llm_calls), "iterations": max(1, report["round"]),
              "reasoning_trace": "", "wall_time_seconds": wall_time,
              "tool_execution_seconds": sum(c["duration_seconds"] for c in calls),
              "total_input_tokens": sum(e["usage"].get("input_tokens", 0) for e in llm_calls),
              "total_output_tokens": sum(e["usage"].get("output_tokens", 0) for e in llm_calls)}
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    return result
