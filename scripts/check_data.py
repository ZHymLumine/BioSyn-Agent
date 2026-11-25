"""Verify packaged task data, case bindings and optional upstream resources."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_path(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Escaping data path: {relative}")
    return path


def check(root=ROOT, require_catalog=False):
    data = root / "data"
    inventory = json.loads((data / "manifest.json").read_text())
    for item in inventory["files"]:
        path = safe_path(data, item["path"])
        if not path.is_file() or sha(path) != item["sha256"]:
            raise ValueError(f"Dataset file missing or changed: {item['path']}")
    recipe = json.loads((data / "cello-catalog-recipe.json").read_text())
    external_catalog = safe_path(root, recipe["path"])
    missing = []
    if external_catalog.exists():
        if sha(external_catalog) != recipe["sha256"]:
            raise ValueError("Cello catalogue differs from its task binding")
    else:
        missing.append(recipe["path"])
    if missing and require_catalog:
        raise ValueError("Fetch the Cello catalogue with: python scripts/fetch_data.py --group catalog")
    datasets = json.loads((data / "datasets.json").read_text())["datasets"]
    total = 0
    for spec in datasets:
        manifest_path = safe_path(data, spec["manifest"])
        directory = manifest_path.parent
        manifest = json.loads(manifest_path.read_text())
        cases = manifest["cases"]
        ids = [row["case_id"] for row in cases]
        if len(ids) != len(set(ids)) or len(ids) != spec["case_count"]:
            raise ValueError(f"Case count or duplicate IDs: {spec['id']}")
        if dict(Counter(row["track"] for row in cases)) != spec["track_counts"]:
            raise ValueError(f"Track counts differ: {spec['id']}")
        for item in manifest.get("files", []):
            path = safe_path(directory, item["path"])
            if path == external_catalog and not path.exists():
                continue
            if sha(path) != item["sha256"]:
                raise ValueError(f"Original dataset binding changed: {item['path']}")
        for row in cases:
            for field in ("task_path", "profile_path", "resources_path", "resource_lock_path",
                          "catalog_path", "reference_path"):
                if field in row:
                    path = safe_path(directory, row[field])
                    if path == external_catalog and not path.exists():
                        continue
                    if not path.is_file():
                        raise ValueError(f"Missing {field}: {row['case_id']}")
            task = json.loads((directory / row["task_path"]).read_text())
            if task["case_id"] != row["case_id"] or task["track"] != row["track"]:
                raise ValueError(f"Task identity mismatch: {row['case_id']}")
            for item in task["inputs"].get("public_resources", []):
                path = safe_path(directory / "public", item["path"])
                if path == external_catalog and not path.exists():
                    continue
                if sha(path) != item["sha256"]:
                    raise ValueError(f"Public resource differs: {row['case_id']}")
        total += len(cases)
    origins = json.loads((data / "provenance/file-origins.json").read_text())
    for item in origins["files"]:
        if sha(safe_path(data, item["path"])) != item["sha256"]:
            raise ValueError(f"Source task bytes changed: {item['path']}")
    return {"cases": total, "bundled_files_verified": len(inventory["files"]),
            "source_files_unchanged": len(origins["files"]), "catalogue_downloads_needed": missing}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--require-catalog", action="store_true")
    args = parser.parse_args()
    print(json.dumps(check(args.root.resolve(), args.require_catalog), indent=2))


if __name__ == "__main__":
    main()
