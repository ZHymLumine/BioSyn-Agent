"""Post-hoc evaluators that were not available to the design agents.

The suite deliberately keeps the formal benchmark score unchanged.  Each track
records the kind of independence it can support: an orthogonal sequence model,
an updated host reconstruction, or a robustness analysis using held-back UCF
cytometry distributions.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from biosyn.applications.computational_circuit import CircuitEvaluator
from biosyn.applications.computational_metabolic import MetabolicEvaluator
from biosyn.applications.computational_sequence import RiboNNPredictor, SalukiPredictor
from biosyn.applications.goal_contract import assess_goals


VERSION = "external-evaluation-v1-20261107"


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "p05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.50)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def _desired_direction(task: dict[str, Any]) -> int:
    """Return the direction declared by the public task, not by a model output."""
    requirements = task["evaluation_contract"]["requirements"]
    preferred = []
    for metric in ("predicted_mrl_change", "calibrated_predicted_mrl"):
        preferred = [row for row in requirements if row["metric"] == metric]
        if preferred:
            break
    if len(preferred) != 1:
        raise ValueError("Expected one public mammalian translation direction requirement")
    operator = preferred[0]["operator"]
    if operator in {">", ">="}:
        return 1
    if operator in {"<", "<="}:
        return -1
    raise ValueError("Equality does not declare an optimization direction")


class CircuitCytometryRobustness:
    """Propagate the UCF's held-back gate cytometry histograms through a circuit."""

    def __init__(self, catalog_path: Path):
        self.catalog_path = Path(catalog_path)
        self.base = CircuitEvaluator(self.catalog_path)
        data = json.loads(self.catalog_path.read_text())
        records = sum(data["sections"].values(), [])
        self.functions = {
            row["name"]: row
            for row in records
            if row.get("collection") == "functions" and "name" in row
        }

    def _table(self, part: str) -> list[dict[str, Any]]:
        gate = self.base.gates[part]
        model = self.base.models[gate["model"]]
        name = model["functions"].get("cytometry")
        if not name or name not in self.functions:
            raise ValueError(f"No UCF cytometry function for gate {part}")
        table = self.functions[name].get("table")
        if not table:
            raise ValueError(f"Empty UCF cytometry table for gate {part}")
        return table

    @staticmethod
    def _sample_table(
        x: np.ndarray, table: list[dict[str, Any]], rng: np.random.Generator
    ) -> np.ndarray:
        table_x = np.asarray([float(row["x"]) for row in table], dtype=float)
        if np.any(table_x <= 0) or np.any(x <= 0):
            raise ValueError("Cytometry interpolation requires positive RPU values")
        nearest = np.abs(np.log10(x)[:, None] - np.log10(table_x)[None, :]).argmin(axis=1)
        out = np.empty_like(x, dtype=float)
        for index in np.unique(nearest):
            mask = nearest == index
            row = table[int(index)]
            bins = np.asarray(row["bin"], dtype=float)
            weights = np.asarray(row["output"], dtype=float)
            if len(bins) != len(weights) or np.any(weights < 0) or weights.sum() <= 0:
                raise ValueError("Malformed UCF cytometry histogram")
            out[mask] = rng.choice(bins, size=int(mask.sum()), p=weights / weights.sum())
        return out

    def evaluate(
        self, task: dict[str, Any], design: dict[str, Any], draws: int, seed: int
    ) -> dict[str, Any]:
        if draws < 100:
            raise ValueError("At least 100 Monte Carlo draws are required")
        nominal = self.base.evaluate(task, design)
        rng = np.random.default_rng(seed)
        states = []
        for truth in task["inputs"]["truth_table"]:
            logic = dict(truth["inputs"])
            values = {
                name: np.full(
                    draws,
                    self.base.sensors[name]["parameters"]["ymax" if value else "ymin"],
                    dtype=float,
                )
                for name, value in logic.items()
            }
            for gate in design["gates"]:
                x = sum(values[source] for source in gate["inputs"])
                values[gate["id"]] = self._sample_table(x, self._table(gate["part"]), rng)
                logic[gate["id"]] = int(not any(logic[source] for source in gate["inputs"]))
            output = values[design["output"]]
            states.append(
                {
                    "inputs": truth["inputs"],
                    "target": truth["outputs"]["output_YFP"],
                    "output_rpu": output,
                }
            )
        on = np.stack([row["output_rpu"] for row in states if row["target"]]).min(axis=0)
        off = np.stack([row["output_rpu"] for row in states if not row["target"]]).max(axis=0)
        ratio = on / off
        state_summary = [
            {
                "inputs": row["inputs"],
                "target": row["target"],
                "output_rpu": _quantiles(row["output_rpu"]),
            }
            for row in states
        ]
        return {
            "status": "complete",
            "independence_level": "robustness_evaluator_same_ucf_data",
            "metrics": {
                "draws": draws,
                "empirical_cytometry_separation_probability": float(np.mean(ratio > 1.0)),
                "on_off_ratio": _quantiles(ratio),
                "log10_on_off_ratio": _quantiles(np.log10(ratio)),
                "minimum_on_rpu": _quantiles(on),
                "maximum_off_rpu": _quantiles(off),
                "nominal_on_off_ratio": nominal["metrics"]["on_off_ratio"],
            },
            "state_distributions": state_summary,
            "scope": (
                "Monte Carlo propagation of the Eco1C1G1T1 empirical gate cytometry histograms. "
                "Input sensors remain at their declared ymin/ymax endpoints; each gate uses the nearest "
                "log-RPU cytometry row. This tests robustness within the Cello UCF and is not independent "
                "experimental validation."
            ),
        }


class ExternalEvaluationSuite:
    def __init__(self, root: Path, dataset: Path, draws: int):
        self.root = Path(root).resolve()
        self.dataset = Path(dataset).resolve()
        self.draws = draws
        self.circuit = CircuitCytometryRobustness(
            self.dataset / "public/resources/cello_Eco1C1G1T1.json"
        )
        self.ribonn: RiboNNPredictor | None = None
        self.saluki: SalukiPredictor | None = None
        self.metabolic: dict[str, MetabolicEvaluator] = {}

    def evaluate_circuit(
        self, task: dict[str, Any], design: dict[str, Any], design_sha256: str
    ) -> dict[str, Any]:
        seed = int(hashlib.sha256((VERSION + design_sha256).encode()).hexdigest()[:16], 16)
        return self.circuit.evaluate(task, design, self.draws, seed)

    @staticmethod
    def evaluate_bacterial(task: dict[str, Any], design: dict[str, Any]) -> dict[str, Any]:
        from ostir import run_ostir

        inputs = task["inputs"]

        def predict(sd: str) -> dict[str, Any]:
            leader = inputs["upstream_before_sd"] + sd + inputs["spacing_sequence"]
            sequence = leader + inputs["fixed_cds"]
            start = len(leader) + 1
            rows = run_ostir(
                sequence,
                start=start,
                end=start,
                threads=1,
                aSD="ACCUCCUUA",
                decimal_places=8,
            )
            if len(rows) != 1 or rows[0]["start_position"] != start:
                raise RuntimeError("OSTIR did not return exactly the annotated start codon")
            return rows[0]

        baseline = predict(inputs["baseline_sd"])
        proposed = predict(design["sd"])
        if baseline["expression"] <= 0 or proposed["expression"] <= 0:
            raise ValueError("OSTIR returned a non-positive translation initiation rate")
        # EMOPEC's target is on its own scale.  The primary model is used only
        # to declare whether the public task asks for an increase or decrease.
        from biosyn.applications.computational_regulatory import RegulatoryEvaluator

        catalog = task["_catalog_path"]
        primary = RegulatoryEvaluator(task, catalog, task["_resource_root"])
        baseline_primary = primary.evaluate(task, {"sd": inputs["baseline_sd"]})["metrics"]
        desired = _sign(inputs["target_relative_activity"] - baseline_primary["emopec_relative_activity"])
        observed = _sign(float(proposed["expression"]) - float(baseline["expression"]))
        return {
            "status": "complete",
            "independence_level": "orthogonal_sequence_model",
            "metrics": {
                "baseline_ostir_tir": float(baseline["expression"]),
                "design_ostir_tir": float(proposed["expression"]),
                "ostir_tir_ratio": float(proposed["expression"] / baseline["expression"]),
                "ostir_log2_tir_ratio": math.log2(
                    float(proposed["expression"] / baseline["expression"])
                ),
                "desired_direction": desired,
                "observed_direction": observed,
                "direction_agreement": observed == desired,
            },
            "raw": {"baseline": baseline, "design": proposed},
            "scope": (
                "OSTIR thermodynamic translation-initiation prediction at the annotated start codon. "
                "Direction is compared with the public EMOPEC target relative to its baseline; numerical "
                "OSTIR and EMOPEC scales are not equated. This is not measured protein expression."
            ),
        }

    def evaluate_mammalian(
        self, task: dict[str, Any], design: dict[str, Any]
    ) -> dict[str, Any]:
        inputs = task["inputs"]
        cds = inputs["fixed_cds"]
        if not inputs["fixed_suffix"].startswith(cds):
            raise ValueError("The formal fixed suffix does not begin with the declared CDS")
        suffix = inputs["fixed_suffix"][len(cds) :] + "A" * inputs["polyA_tail_nt"]
        design_prefix = inputs["fixed_leader"] + design["utr"]
        baseline_prefix = inputs["fixed_leader"] + inputs["baseline_utr"]
        assembled = design_prefix + cds + suffix
        if design["rna_body"].replace("U", "T") != assembled:
            raise ValueError("Submitted RNA does not match the public fixed-context assembly")
        resources = self.root / "resources/application_suite_v3/compute_models"
        if self.ribonn is None:
            self.ribonn = RiboNNPredictor(resources / "ribonn")
        if self.saluki is None:
            self.saluki = SalukiPredictor(resources / "saluki")
        baseline_ribonn = self.ribonn.predict(baseline_prefix, cds, suffix)
        design_ribonn = self.ribonn.predict(design_prefix, cds, suffix)
        baseline_saluki = self.saluki.predict(baseline_prefix, cds, suffix)
        design_saluki = self.saluki.predict(design_prefix, cds, suffix)
        ribonn_delta = design_ribonn["ribonn_HEK293T_score"] - baseline_ribonn[
            "ribonn_HEK293T_score"
        ]
        desired = _desired_direction(task)
        return {
            "status": "complete",
            "independence_level": "orthogonal_full_transcript_models",
            "metrics": {
                "baseline_ribonn_HEK293T_score": baseline_ribonn["ribonn_HEK293T_score"],
                "design_ribonn_HEK293T_score": design_ribonn["ribonn_HEK293T_score"],
                "ribonn_HEK293T_delta": ribonn_delta,
                "ribonn_model_count": design_ribonn["ribonn_model_count"],
                "baseline_ribonn_model_sd": baseline_ribonn["ribonn_model_sd"],
                "design_ribonn_model_sd": design_ribonn["ribonn_model_sd"],
                "desired_translation_direction": desired,
                "ribonn_direction_agreement": _sign(ribonn_delta) == desired,
                "baseline_saluki_human_score": baseline_saluki,
                "design_saluki_human_score": design_saluki,
                "saluki_human_delta": design_saluki - baseline_saluki,
            },
            "scope": (
                "RiboNN HEK293T translation-efficiency score is the orthogonal translation endpoint. "
                "Saluki supplies a separate human mRNA-stability score and is not counted as translation "
                "agreement. Neither score is calibrated to the Sample reporter's MRL units or a half-life "
                "in hours. The complete fixed leader, eGFP CDS, 3' context and encoded poly(A) sequence are used."
            ),
        }

    def evaluate_metabolic(
        self, case_id: str, task: dict[str, Any], design: dict[str, Any]
    ) -> dict[str, Any]:
        external_task = copy.deepcopy(task)
        conditions = external_task["inputs"]["computational_conditions"]
        conditions.update(
            {
                "model_id": "iML1515",
                "model_path": "tasks/application_suite_v2/sources/iML1515.xml.gz",
                "biomass_reaction": "BIOMASS_Ec_iML1515_core_75p37M",
            }
        )
        if case_id not in self.metabolic:
            self.metabolic[case_id] = MetabolicEvaluator(
                external_task,
                self.dataset / "public/resources/metabolic-library.json",
                self.root,
            )
        result = self.metabolic[case_id].evaluate(external_task, design)
        result.update(assess_goals(external_task, result))
        result.update(
            {
                "status": "complete",
                "independence_level": "updated_host_reconstruction",
                "scope": (
                    "Cross-model FBA in iML1515 under the same declared glucose uptake, medium and "
                    "20% wild-type growth fraction. Heterologous reactions are re-added from the frozen "
                    "curated stoichiometry; every selected native reaction must exist with an exact directed "
                    "stoichiometric match. This remains a steady-state upper bound, not titre or growth coupling."
                ),
                "model": {
                    "id": "iML1515",
                    "path": "tasks/application_suite_v2/sources/iML1515.xml.gz",
                },
            }
        )
        return result


def _rank(values: list[float]) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # Average tied ranks.
    array = np.asarray(values)
    for value in np.unique(array):
        mask = array == value
        ranks[mask] = ranks[mask].mean()
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    rx, ry = _rank(x), _rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in records if row["external_evaluation"].get("status") == "complete"]
    tracks = {}
    for track in sorted({row["track"] for row in records}):
        selected = [row for row in records if row["track"] == track]
        tracks[track] = {
            "scheduled": len(selected),
            "complete": sum(row["external_evaluation"].get("status") == "complete" for row in selected),
            "not_evaluable": sum(
                row["external_evaluation"].get("status") == "not_evaluable" for row in selected
            ),
            "error": sum(row["external_evaluation"].get("status") == "error" for row in selected),
        }
    bacterial = [row for row in complete if row["track"] == "bacterial_cds"]
    mammalian = [row for row in complete if row["track"] == "mammalian_mrna"]
    metabolic = [row for row in complete if row["track"] == "metabolic_pathway"]
    circuits = [row for row in complete if row["track"] == "genetic_circuit"]
    return {
        "version": VERSION,
        "scheduled": len(records),
        "complete": len(complete),
        "tracks": tracks,
        "bacterial": {
            "direction_agreement": sum(
                row["external_evaluation"]["metrics"]["direction_agreement"] for row in bacterial
            ),
            "denominator": len(bacterial),
            "spearman_internal_emopec_vs_external_ostir": _spearman(
                [row["internal_metrics"]["emopec_relative_activity"] for row in bacterial],
                [row["external_evaluation"]["metrics"]["design_ostir_tir"] for row in bacterial],
            ),
        },
        "mammalian": {
            "ribonn_direction_agreement": sum(
                row["external_evaluation"]["metrics"]["ribonn_direction_agreement"]
                for row in mammalian
            ),
            "denominator": len(mammalian),
            "spearman_internal_mrl_delta_vs_ribonn_delta": _spearman(
                [row["internal_metrics"]["predicted_mrl_change"] for row in mammalian],
                [row["external_evaluation"]["metrics"]["ribonn_HEK293T_delta"] for row in mammalian],
            ),
        },
        "metabolic": {
            "iml1515_feasible": sum(
                bool(row["external_evaluation"].get("feasible")) for row in metabolic
            ),
            "iml1515_public_goal_met": sum(
                bool(row["external_evaluation"].get("goal_met")) for row in metabolic
            ),
            "denominator": len(metabolic),
            "spearman_iJO1366_vs_iML1515_yield": _spearman(
                [row["internal_metrics"]["product_yield_mol_per_mol"] for row in metabolic],
                [
                    row["external_evaluation"]["metrics"]["product_yield_mol_per_mol"]
                    for row in metabolic
                ],
            ),
        },
        "genetic_circuit": {
            "denominator": len(circuits),
            "separation_probability": (
                _quantiles(
                    np.asarray(
                        [
                            row["external_evaluation"]["metrics"][
                                "empirical_cytometry_separation_probability"
                            ]
                            for row in circuits
                        ]
                    )
                )
                if circuits
                else None
            ),
            "scope": "robustness within the Cello UCF; not an independent biological model",
        },
        "interpretation": (
            "External metrics are post-hoc secondary evidence. They do not replace or revise the formal "
            "v13 goal_met field. No cross-track aggregate pass/fail score is defined."
        ),
    }


def evaluate_formal_results(
    root: Path,
    release: Path,
    results: Path,
    output: Path,
    draws: int = 5000,
) -> dict[str, Any]:
    root, release, results, output = map(Path, (root, release, results, output))
    root, release, results, output = (p.resolve() for p in (root, release, results, output))
    if output.exists():
        raise FileExistsError(f"External evaluation output already exists: {output}")
    release_spec = json.loads((release / "release.json").read_text())
    matrix = json.loads((release / "matrix.json").read_text())
    dataset = (root / release_spec["dataset"]).resolve()
    manifest = json.loads((dataset / "manifest.json").read_text())
    rows = {row["case_id"]: row for row in manifest["cases"]}
    schedule = [
        (batch["id"], item)
        for batch in matrix["batches"]
        if batch["formal"]
        for item in batch["schedule"]
    ]
    if len(schedule) != matrix["formal_episodes"]:
        raise ValueError("Formal schedule length does not match the release matrix")
    output.mkdir(parents=True)
    provenance = {
        "version": VERSION,
        "formal_release": str(release),
        "formal_release_sha256": _sha256(release / "release.json"),
        "formal_results": str(results),
        "draws": draws,
        "models": {
            "cello_ucf_sha256": _sha256(
                dataset / "public/resources/cello_Eco1C1G1T1.json"
            ),
            "iML1515_sha256": _sha256(
                root / "tasks/application_suite_v2/sources/iML1515.xml.gz"
            ),
            "ribonn_source_manifest_sha256": _sha256(
                root
                / "resources/application_suite_v3/compute_models/ribonn/source-manifest.json"
            ),
            "saluki_checkpoint_sha256": _sha256(
                root / "resources/application_suite_v3/compute_models/saluki/model_best.h5"
            ),
            "ostir_version": importlib.metadata.version("ostir"),
        },
    }
    _write(output / "provenance.json", provenance)
    suite = ExternalEvaluationSuite(root, dataset, draws)
    records = []
    for batch_id, item in schedule:
        case_id, method, seed = item["case_id"], item["method"], item["seed"]
        case = rows[case_id]
        task = json.loads((dataset / case["task_path"]).read_text())
        task["_catalog_path"] = str(dataset / case["catalog_path"])
        task["_resource_root"] = str(root / "resources/application_suite_v3")
        folder = results / "batches" / batch_id / f"{case_id}--{method}--{seed}"
        result_path, evaluation_path, submission_path = (
            folder / "result.json",
            folder / "evaluation.json",
            folder / "submission.json",
        )
        if not result_path.is_file():
            external = {"status": "error", "reason": "formal_result_missing"}
            formal_result, internal_metrics = {}, {}
        else:
            formal_result = json.loads(result_path.read_text())
            internal = json.loads(evaluation_path.read_text()) if evaluation_path.is_file() else {}
            internal_metrics = internal.get("metrics", {})
            if not formal_result.get("design_valid") or not submission_path.is_file():
                external = {
                    "status": "not_evaluable",
                    "reason": formal_result.get("failure_category", "no_valid_submission"),
                }
            else:
                design = json.loads(submission_path.read_text())
                try:
                    if case["track"] == "genetic_circuit":
                        external = suite.evaluate_circuit(
                            task, design, formal_result["final"]["design_sha256"]
                        )
                    elif case["track"] == "bacterial_cds":
                        external = suite.evaluate_bacterial(task, design)
                    elif case["track"] == "mammalian_mrna":
                        external = suite.evaluate_mammalian(task, design)
                    elif case["track"] == "metabolic_pathway":
                        external = suite.evaluate_metabolic(case_id, task, design)
                    else:
                        raise ValueError(f"Unsupported formal track: {case['track']}")
                except Exception as exc:  # retained per episode; aggregate fails visibly below
                    external = {
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    }
        record = {
            "version": VERSION,
            "case_id": case_id,
            "track": case["track"],
            "method": method,
            "seed": seed,
            "formal_goal_met": formal_result.get("goal_met"),
            "formal_design_valid": formal_result.get("design_valid"),
            "internal_metrics": internal_metrics,
            "external_evaluation": external,
        }
        records.append(record)
        _write(output / "episodes" / f"{case_id}--{method}--{seed}.json", record)
    summary = summarize(records)
    _write(output / "records.json", records)
    _write(output / "summary.json", summary)
    if any(row["external_evaluation"].get("status") == "error" for row in records):
        raise RuntimeError("One or more external evaluations failed; inspect records.json")
    return summary
