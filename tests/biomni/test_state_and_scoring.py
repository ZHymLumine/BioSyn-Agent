import math

import pytest
from pydantic import ValidationError

from biosyn.agent.optimizer import score, select
from biosyn.agent.state import Candidate, Evaluation, Objective, RunStore, TaskSpec


def objective(**kwargs):
    return Objective(metric="x", direction="maximize", lower=0, upper=10, minimum=5, **kwargs)


@pytest.mark.parametrize("values", [{"dna": "atgc"}, {"rna": "ATGC"}, {"protein": "MA*"}, {"dna": ""}, {}])
def test_invalid_sequence_is_not_silently_repaired(values):
    with pytest.raises(ValidationError):
        Candidate(id="invalid", **values)


def test_threshold_and_direction():
    value, violations = score({"x": 4}, [objective()])
    assert value == 0.4 and violations
    assert score({"x": 5}, [objective()]) == (0.5, [])
    o = Objective(metric="loss", direction="minimize", lower=-10, upper=0, maximum=-2)
    assert score({"loss": -8}, [o]) == (0.8, [])


def test_missing_metric_cannot_pass():
    with pytest.raises(KeyError):
        score({}, [objective()])


@pytest.mark.parametrize("number", [math.nan, math.inf, -math.inf])
def test_nonfinite_metrics_rejected(number):
    with pytest.raises(ValidationError):
        Evaluation(candidate_id="bad", metrics={"x": number})


def test_selection_excludes_infeasible_high_score():
    results = {"a": Evaluation(candidate_id="a", metrics={"x": 6}, score=0.6, passed=True),
               "b": Evaluation(candidate_id="b", metrics={"x": 9}, score=0.9, passed=False)}
    assert select(results, [objective()]) == ("a", ["a"])


def test_pareto_keeps_tradeoffs():
    objectives = [objective(), Objective(metric="y", direction="minimize", lower=0, upper=10)]
    results = {name: Evaluation(candidate_id=name, metrics={"x": x, "y": y}, score=0.5, passed=True)
               for name, x, y in (("a", 9, 8), ("b", 7, 3), ("c", 6, 9))}
    assert select(results, objectives)[1] == ["a", "b"]


def test_exclusive_run_directory(tmp_path):
    RunStore(tmp_path / "run")
    with pytest.raises(FileExistsError):
        RunStore(tmp_path / "run")


def test_task_rejects_unmeasurable_objective():
    with pytest.raises(ValidationError, match="Every objective"):
        TaskSpec(id="test", request="test", objectives=[objective()], evaluators=[
            {"name": "stats", "tool": "dna.sequence_analyze", "inputs": {"dna": "dna"}, "metrics": {"gc": "gc_fraction"}}])
