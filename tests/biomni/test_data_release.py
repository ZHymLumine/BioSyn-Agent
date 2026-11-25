"""Data publication checks and exact-byte download behavior."""

import hashlib
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_formal_case_bindings():
    report = load("check_data").check(ROOT)
    assert report["cases"] == 22
    assert report["source_files_unchanged"] >= 120


def test_download_rejects_corrupt_payload(tmp_path):
    target = tmp_path / "source.json"
    with pytest.raises(ValueError, match="SHA-256"):
        load("fetch_data").store(target, b"changed", hashlib.sha256(b"original").hexdigest())
    assert not target.exists()


def test_download_preserves_existing_different_file(tmp_path):
    target = tmp_path / "source.json"
    target.write_bytes(b"local edit")
    with pytest.raises(ValueError, match="Existing file differs"):
        load("fetch_data").store(target, b"original", hashlib.sha256(b"original").hexdigest())
    assert target.read_bytes() == b"local edit"


def test_manifest_path_stays_inside_root(tmp_path):
    with pytest.raises(ValueError, match="Path outside"):
        load("fetch_data").within(tmp_path, "../outside.json")
