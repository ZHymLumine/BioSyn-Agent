import json
from biosyn.applications.comparison import select_fixed, execute_record
from biosyn.applications.computational_metabolic import routes_to_product


def test_growth_feasibility_precedes_metabolic_yield():
    history = [
        {"evaluation": {"valid": True, "feasible": False, "metrics": {"yield": 2}}},
        {"evaluation": {"valid": True, "feasible": True, "metrics": {"yield": 1}}},
    ]
    assert select_fixed(history, [{"metric": "yield", "direction": "maximize"}]) is history[1]


def test_pathway_enumeration_stops_cycles_and_requires_target():
    task = {"inputs": {"product_metabolite": "p", "central_precursors": ["s"]}}
    resources = {
        "reactions": {
            "start": {"main_substrate": "s", "main_product": "a"},
            "cycle": {"main_substrate": "a", "main_product": "s"},
            "end": {"main_substrate": "a", "main_product": "p"},
            "unrelated": {"main_substrate": "x", "main_product": "p"},
        }
    }
    assert routes_to_product(task, resources) == [["start", "end"]]


def test_resume_preserves_existing_results_and_interrupted_attempts(tmp_path):
    row = {"case_id": "C1"}
    case = tmp_path / "C1--biosyn--101"
    case.mkdir()
    (case / "evidence.txt").write_text("original partial attempt")
    failed = execute_record(tmp_path, tmp_path, tmp_path, row, "biosyn", 101, {}, "")
    assert failed["failure_category"] == "infrastructure_interruption"
    assert (case / "evidence.txt").read_text() == "original partial attempt"
    assert execute_record(tmp_path, tmp_path, tmp_path, row, "biosyn", 101, {}, "") == failed
    assert json.loads((case / "result.json").read_text()) == failed
