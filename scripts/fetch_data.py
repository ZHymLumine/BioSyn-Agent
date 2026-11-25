"""Fetch pinned task resources, evaluator assets or original source tables."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]


def digest(content):
    return hashlib.sha256(content).hexdigest()


def within(root, relative):
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root.resolve()):
        raise ValueError(f"Path outside repository: {relative}")
    return path


def store(path, content, expected):
    if digest(content) != expected:
        raise ValueError(f"SHA-256 mismatch: {path.name}")
    if path.exists():
        if digest(path.read_bytes()) != expected:
            raise ValueError(f"Existing file differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(content)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def fetch(root, groups):
    manifest = json.loads((root / "data/external-resources.json").read_text())
    count = 0
    for entry in manifest["resources"]:
        if entry["group"] not in groups:
            continue
        path = within(root, entry["destination"])
        if path.exists():
            content = path.read_bytes()
        else:
            if not entry["url"].startswith("https://"):
                raise ValueError("Expected an HTTPS resource URL")
            request = Request(entry["url"], headers={"User-Agent": "BioSyn-Agent-data"})
            with urlopen(request, timeout=240) as response:
                content = response.read()
            if digest(content) != entry.get("download_sha256", entry["sha256"]):
                raise ValueError(f"Downloaded source SHA-256 mismatch: {entry['id']}")
            if "transform" in entry:
                if entry["transform"] != "remove_leading_utf8_encoding_cookie":
                    raise ValueError(f"Unsupported resource transformation: {entry['transform']}")
                header, content = content.split(b"\n", 1)
                if header.rstrip(b"\r") != b"# -*- coding: utf-8 -*-":
                    raise ValueError("Expected the recorded leading UTF-8 declaration")
        store(path, content, entry["sha256"])
        count += 1
        print(f"Verified {entry['id']}", flush=True)
    if "evaluation" in groups:
        for entry in manifest["local_copies"]:
            content = within(root, entry["source"]).read_bytes()
            store(within(root, entry["destination"]), content, entry["sha256"])
    if "catalog" in groups:
        recipe = json.loads((root / "data/cello-catalog-recipe.json").read_text())
        catalog = recipe["template"]
        for section in catalog["sections"]:
            content = (root / f"resources/application_suite_v3/cello/{section}.json").read_bytes()
            excluded = recipe.get("excluded_collections", {}).get(section, [])
            catalog["sections"][section] = [row for row in json.loads(content)
                                               if row.get("collection") not in excluded]
        content = (json.dumps(catalog, ensure_ascii=False, indent=2) + "\n").encode()
        store(within(root, recipe["path"]), content, recipe["sha256"])
        print("Reconstructed Cello catalogue matches the task-bound SHA-256.")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--group", nargs="+", choices=["catalog", "evaluation", "source-tables"],
                        default=["catalog"])
    args = parser.parse_args()
    count = fetch(args.root.resolve(), set(args.group))
    print(f"Verified {count} upstream files.")


if __name__ == "__main__":
    main()
