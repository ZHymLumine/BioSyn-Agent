"""Verify explicit file and package bindings before starting a scientific evaluator."""

import hashlib
import importlib.metadata
import json
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_resources(root, lock_path, track):
    root = Path(root).resolve()
    raw = Path(lock_path).read_bytes()
    lock = json.loads(raw)
    if lock["track"] != track or not lock["files"]:
        raise ValueError("Scientific resource lock does not match the task")
    for entry in lock["files"]:
        path = (root / entry["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError("Missing or escaping scientific resource: " + entry["path"])
        if digest(path) != entry["sha256"]:
            raise ValueError("Scientific resource changed: " + entry["path"])
    for package, version in lock["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("Scientific package version changed: " + package)
    return {
        "resource_lock_sha256": hashlib.sha256(raw).hexdigest(),
        "verified_file_count": len(lock["files"]),
        "packages": lock["packages"],
    }
