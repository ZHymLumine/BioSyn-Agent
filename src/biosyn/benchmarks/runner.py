"""Protocol-coverage smoke tests; preserve original task files and grade only after isolation ends."""
import argparse
import copy
import gzip
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import dotenv_values

from biosyn.benchmarks.proxy import deepseek_proxy


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def gzrows(path):
    with gzip.open(path, "rt") as handle:
        return [json.loads(line) for line in handle]


def choose(rows):
    # Selection is defined before any submission or label is read.
    return min(rows, key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest())


def select_cases(root):
    result = []
    cases = root / "tasks/circuit_reasoning/cases"
    rows = gzrows(cases / "tasks.jsonl.gz")
    for kind in ("T1", "T2", "T3", "T4", "T5", "T6"):
        result.append(("circuit", kind, choose([r for r in rows if r["task_type"] == kind]), cases))
    for mode in ("functional", "topology_realization"):
        result.append(("circuit", "T7_" + mode, choose([r for r in rows if r["task_type"] == "T7"
                                                       and r["design_mode"] == mode]), cases))
    expansion = root / "tasks/circuit_reasoning/expansion/cases"
    result.append(("circuit", "T7_expansion", choose(gzrows(expansion / "tasks.jsonl.gz")), expansion))
    result.append(("circuit", "E1", choose(gzrows(cases / "expression_tasks.jsonl.gz")), cases))
    application = root / "tasks/application_suite_v2/cases"
    rows = [json.loads(p.read_text()) for p in sorted(application.glob("*/task.json"))]
    for kind in ("expression_construct_v2", "expression_diagnosis", "metabolic_pathway_v2"):
        eligible = [r for r in rows if r["task_type"] == kind and r["split"] == "development"]
        result.append(("application", kind, choose(eligible), application))
    return result


def stage(root, case_dir, selected):
    suite, category, original, source = selected
    public = case_dir / "public"
    work = case_dir / "work"
    public.mkdir(parents=True)
    work.mkdir()
    task = copy.deepcopy(original)
    # Legacy diagnosis ids contain the fault level. Do not expose this name to the model.
    task["id"] = "episode_" + hashlib.sha256(original["id"].encode()).hexdigest()[:16]
    write(public / "task.json", task)
    if suite == "circuit":
        protocol_name = "expression_protocol.md" if category == "E1" else "protocol.md"
        # The expansion distributes tasks/labels only and uses the core public contract/library.
        common = root / "tasks/circuit_reasoning/cases"
        (public / "protocol.md").write_bytes((common / protocol_name).read_bytes())
        if category == "E1":
            catalog = json.loads((source / "expression_catalog.json").read_text())
            allowed = set(task["candidate_promoter_ids"]) | {task["reference_promoter_id"]}
            catalog["parts"] = [p for p in catalog["parts"] if p["id"] in allowed]
            catalog["backgrounds"] = [b for b in catalog["backgrounds"] if b["id"] == task["background_id"]]
            write(public / "expression_catalog.json", catalog)
        else:
            (public / "parts_library.json").write_bytes((common / "parts_library.json").read_bytes())
    else:
        (public / "protocol.md").write_text(
            "Application suite compatibility smoke test. Read task.json for the unchanged public goal and spec. "
            "Original grading modules are absent from this server checkout. The submit tool checks the "
            "public candidate structure and limited sequence invariants only. Pursue the goal using available "
            "tools, state unresolved objectives, and do not claim task success or measured phenotype.\n")
    answer_format = original.get("answer_format", "json")
    packet = {"suite": suite, "category": category, "task": task, "answer_format": answer_format,
              "server_root": str(root), "private_feedback": False}
    write(public / "packet.json", packet)
    config = {
        "servers": {"dna": {"command": [str(root / ".venv-biomni/bin/python"), "-m", "biosyn.mcp.dna"]},
                    "rna": {"command": [str(root / ".venv-biomni/bin/python"), "-m", "biosyn.mcp.rna"]}},
        "llm": {"model": "deepseek-v4-flash", "base_url": "http://deepseek.local/v1",
                "api_key_env": "BIOSYN_PROXY_KEY", "api_socket": "/proxy/llm.sock",
                "max_tokens": 8192, "timeout": 240, "temperature": 0.2,
                "extra_body": {"thinking": {"type": "disabled"}}},
        "biomni_tool_modules": [], "use_tool_retriever": True, "recursion_limit": 48, "code_timeout": 120,
    }
    write(public / "config.json", config)
    return public, work


def command(root, public, work, socket_path):
    prefix = Path(sys.base_prefix)
    cmd = ["bwrap", "--die-with-parent", "--new-session", "--unshare-user", "--unshare-pid",
           "--unshare-ipc", "--unshare-uts", "--unshare-net", "--cap-drop", "ALL"]
    mounts = [(Path("/usr"), Path("/usr")), (Path("/bin"), Path("/bin")), (Path("/lib64"), Path("/lib64")),
              (Path("/lib"), Path("/lib")), (prefix / "bin", prefix / "bin"),
              (prefix / "lib", prefix / "lib"), (root / ".venv-biomni", root / ".venv-biomni"),
              (root / "Biomni", root / "Biomni"), (root / "src", root / "src")]
    for src, dst in mounts:
        cmd += ["--ro-bind", str(src), str(dst)]
    cmd += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
            "--ro-bind", str(public), "/public", "--bind", str(work), "/work",
            "--ro-bind", str(socket_path), "/proxy/llm.sock",
            "--setenv", "HOME", "/work/home", "--setenv", "PATH", str(root / ".venv-biomni/bin") + ":/usr/bin",
            "--setenv", "BIOSYN_PROXY_KEY", "local-transport-only", "--setenv", "PYTHONUNBUFFERED", "1",
            "--setenv", "OMP_NUM_THREADS", "2", "--setenv", "OPENBLAS_NUM_THREADS", "2",
            "--chdir", "/work", str(root / ".venv-biomni/bin/python"), "-m", "biosyn.benchmarks.worker"]
    return cmd


def grade_circuit(root, selected, artifact, output_dir):
    _, category, original, source = selected
    directory = root / "tasks/circuit_reasoning"
    script = "expression_grade.py" if category == "E1" else "grade.py"
    grading_data = source.parent
    if category == "T7_expansion":
        # Keep the upstream grader unmodified; supply its expected directory layout.
        grading_data = output_dir / "grader-data"
        grading_data.mkdir(mode=0o700)
        (grading_data / "cases").symlink_to(directory / "cases", target_is_directory=True)
        (grading_data / "labels").symlink_to(source.parent / "labels", target_is_directory=True)
    command_line = [str(root / ".venv-biomni/bin/python"), str(directory / script),
                    original["id"], str(artifact), "--data", str(grading_data)]
    proc = subprocess.run(command_line, cwd=directory, capture_output=True, text=True, timeout=90)
    (output_dir / "grader.stderr").write_text(proc.stderr)
    (output_dir / "grader.stdout").write_text(proc.stdout)
    if proc.returncode:
        return {"grading_status": "error", "returncode": proc.returncode, "error": proc.stderr[-3000:]}
    return {"grading_status": "graded", **json.loads(proc.stdout)}


def run_case(root, output, selected, api_key, wall_seconds):
    suite, category, original, _ = selected
    case_dir = output / category
    public, work = stage(root, case_dir, selected)
    start = time.monotonic()
    result = {"suite": suite, "category": category, "task_id": original["id"],
              "public_task_sha256": hashlib.sha256((public / "task.json").read_bytes()).hexdigest(),
              "original_task_sha256": hashlib.sha256(json.dumps(original, sort_keys=True).encode()).hexdigest()}
    with tempfile.TemporaryDirectory(prefix="biosyn-proxy-") as temp:
        socket_path = Path(temp) / "llm.sock"
        with deepseek_proxy(socket_path, api_key, case_dir / "api-usage.jsonl") as proxy:
            cmd = command(root, public, work, socket_path)
            write(case_dir / "execution.json", {"command": cmd, "wall_seconds": wall_seconds,
                                               "llm_call_limit": 24, "max_output_tokens": 8192})
            with (case_dir / "agent.log").open("w") as log:
                proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"})
                try:
                    result["process_returncode"] = proc.wait(timeout=wall_seconds)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
                    result["process_returncode"] = proc.returncode
                    result["timeout"] = True
            result["api_calls"] = proxy["calls"]
    episode = work / "output"
    receipt_file = episode / "submission-receipt.json"
    if receipt_file.exists():
        receipt = json.loads(receipt_file.read_text())
        artifact = episode / receipt["artifact"]
        if hashlib.sha256(artifact.read_bytes()).hexdigest() != receipt["sha256"]:
            raise ValueError("Final submission differs from accepted artifact")
        result["submission"] = receipt
        if suite == "circuit":
            result["grade"] = grade_circuit(root, selected, artifact, case_dir)
        else:
            result["grade"] = {"grading_status": "unavailable", "success": None,
                               "reason": "Original application grading modules are absent; user authorized format checks only."}
    else:
        result["submission"] = None
        result["grade"] = {"grading_status": "not_submitted", "success": False if suite == "circuit" else None}
    result["seconds"] = round(time.monotonic() - start, 2)
    write(case_dir / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--categories", nargs="*")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--wall-seconds", type=int, default=900)
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    key = dotenv_values(args.env_file).get("DEEPSEEK_API_KEY")
    if not key:
        raise ValueError("DEEPSEEK_API_KEY absent from authorized credential file")
    selected = select_cases(root)
    if args.categories:
        unknown = set(args.categories) - {x[1] for x in selected}
        if unknown:
            raise ValueError(f"Unknown categories {unknown}")
        selected = [x for x in selected if x[1] in args.categories]
    write(output / "selection.json", {"selection_rule": "lowest SHA256(task id) per protocol category; application development split only",
          "scope": "protocol-coverage smoke, not a full benchmark or model comparison",
          "cases": [{"suite": s, "category": c, "task_id": t["id"]} for s, c, t, _ in selected]})
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_case, root, output, item, key, args.wall_seconds): item for item in selected}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"suite": item[0], "category": item[1], "task_id": item[2]["id"],
                          "harness_error": f"{type(exc).__name__}: {exc}"}
            results.append(result)
            write(output / "summary.json", results)
            print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
