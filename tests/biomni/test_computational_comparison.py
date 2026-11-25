import itertools
import json

import pytest

from biosyn.applications.comparison import select_fixed
from biosyn.applications.comparison_validation import compare_values
from biosyn.applications.computational_circuit import synthesize_nor
from biosyn.applications.computational_resources import digest, verify_resources


def test_generic_nor_synthesis_all_nonconstant_three_input_functions():
    states = list(itertools.product([0, 1], repeat=3))
    for mask in range(1, 255):
        truth = [
            {"inputs": dict(zip(["a", "b", "c"], state)), "outputs": {"output_YFP": (mask >> index) & 1}}
            for index, state in enumerate(states)
        ]
        graph = synthesize_nor(truth)
        for row in truth:
            value = dict(row["inputs"])
            for gate in graph["gates"]:
                assert 1 <= len(gate["inputs"]) <= 2
                value[gate["id"]] = int(not any(value[x] for x in gate["inputs"]))
            assert value[graph["output"]] == row["outputs"]["output_YFP"]


def test_resource_binding_detects_changed_bytes_and_wrong_track(tmp_path):
    source = tmp_path / "model.bin"
    source.write_bytes(b"original")
    lock = tmp_path / "lock.json"
    lock.write_text(
        json.dumps(
            {
                "track": "bacterial_cds",
                "packages": {},
                "files": [{"path": "model.bin", "sha256": digest(source)}],
            }
        )
    )
    assert verify_resources(tmp_path, lock, "bacterial_cds")["verified_file_count"] == 1
    with pytest.raises(ValueError, match="does not match"):
        verify_resources(tmp_path, lock, "mammalian_mrna")
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        verify_resources(tmp_path, lock, "bacterial_cds")


def test_resource_binding_rejects_path_outside_root(tmp_path):
    inner = tmp_path / "root"
    inner.mkdir()
    source = tmp_path / "outside"
    source.write_bytes(b"data")
    lock = inner / "lock.json"
    lock.write_text(
        json.dumps(
            {"track": "x", "packages": {}, "files": [{"path": "../outside", "sha256": digest(source)}]}
        )
    )
    with pytest.raises(ValueError, match="escaping"):
        verify_resources(inner, lock, "x")


def test_fixed_selection_hard_logic_precedes_analog_score():
    history = [
        {
            "candidate": {"id": "bad_logic"},
            "evaluation": {"valid": True, "metrics": {"truth_table_accuracy": 0.875, "ratio": 100}},
        },
        {
            "candidate": {"id": "correct_logic"},
            "evaluation": {"valid": True, "metrics": {"truth_table_accuracy": 1.0, "ratio": 0.2}},
        },
    ]
    assert (
        select_fixed(history, [{"metric": "ratio", "direction": "maximize"}])["candidate"]["id"]
        == "correct_logic"
    )


def test_rescoring_records_changed_metric_and_artifact():
    original = {"metrics": {"score": 1.0}, "artifact": {"cds": "ATGTAA"}}
    assert not compare_values(original, {"metrics": {"score": 1.00000001}, "artifact": {"cds": "ATGTAA"}})
    assert compare_values(original, {"metrics": {"score": 2.0}, "artifact": {"cds": "ATGTAA"}})
    assert compare_values(original, {"metrics": {"score": 1.0}, "artifact": {"cds": "ATGTAG"}})
