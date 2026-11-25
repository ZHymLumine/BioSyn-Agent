"""Bind the archived formal tasks to a new checkout and verified scientific assets."""

import argparse
import json
import shutil
from pathlib import Path

from check_data import ROOT, check, safe_path, sha


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def prepare(root, output):
    check(root, require_catalog=True)
    source = root / "data/formal_v2_expanded_20261102"
    old_prefix = "tasks/application_suite_v3/formal_v2_expanded_20261102/"
    manifest = json.loads((source / "manifest.json").read_text())
    if output.exists() or not output.is_relative_to(root):
        raise ValueError("Choose a new output directory inside the checkout")
    bindings = []
    locks = {}
    for case in manifest["cases"]:
        old = json.loads((source / case["resource_lock_path"]).read_text())
        entries = []
        for item in old["files"]:
            name = item["path"]
            if name.startswith(old_prefix):
                relative = name.removeprefix(old_prefix)
                actual = safe_path(source, relative)
                target = safe_path(output, relative)
                if sha(actual) != item["sha256"]:
                    raise ValueError(f"Scientific input changed: {name}")
            elif name.startswith("resources/application_suite_v3/"):
                target = safe_path(root, name)
                if not target.exists() or sha(target) != item["sha256"]:
                    raise ValueError(f"Fetch the pinned evaluator asset first: {name}")
                actual = target
            elif name.startswith("runs/") and "/runtime/src/" in name:
                target = safe_path(root, "src/" + name.split("/runtime/src/", 1)[1])
                actual = target
            else:
                raise ValueError(f"Unrecognized historical binding: {name}")
            entry = {"path": str(target.relative_to(root)), "sha256": sha(actual)}
            entries.append(entry)
            bindings.append({"case_id": case["case_id"], "original": item, "prepared": entry})
        locks[case["resource_lock_path"]] = {"track": old["track"], "files": entries,
                                               "packages": old["packages"]}
    shutil.copytree(source, output)
    for name, value in locks.items():
        write(output / name, value)
    for item in manifest["files"]:
        item["sha256"] = sha(output / item["path"])
    manifest["prepared_from"] = {"path": str(source.relative_to(root)),
                                  "manifest_sha256": sha(source / "manifest.json")}
    manifest["version"] += "-prepared"
    write(output / "manifest.json", manifest)
    write(output / "preparation.json", {"bindings": bindings,
          "scientific_task_profile_reference_bytes": "unchanged",
          "runtime_binding": "current checkout; historical bindings retained in archived data"})
    return {"dataset": str(output), "cases": len(manifest["cases"]),
            "resource_locks": len(locks)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    print(json.dumps(prepare(root, output), indent=2))


if __name__ == "__main__":
    main()
