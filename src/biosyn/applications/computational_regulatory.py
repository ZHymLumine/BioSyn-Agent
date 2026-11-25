"""Source-bound RBS and 5'UTR design; numerical endpoints retain their model scope."""

import importlib
import json
import math
import sys
from pathlib import Path

from biosyn.applications.science import Optimus, sequence


class RegulatoryEvaluator:
    def __init__(self, task, catalog_path, resource_root):
        self.catalog = json.loads(Path(catalog_path).read_text())
        self.root = Path(resource_root)
        self.engine = task["evaluation_engine"]
        self.model = None
        if self.engine == "emopec_sd":
            sys.path.insert(0, str(self.root / self.catalog["software_path"]))
            self.emopec = importlib.import_module("emopec._emopec")
        elif self.engine == "optimus_utr":
            self.calibration = json.loads((self.root / self.catalog["calibration_path"]).read_text())
        else:
            raise ValueError("Unsupported regulatory evaluator")

    def materialize(self, task, design):
        if not isinstance(design, dict):
            raise ValueError("Design must be an object")
        i = task["inputs"]
        if self.engine == "emopec_sd":
            if set(design) - {"sd", "upstream_window", "cds", "dna_sequence", "model_window"} or "sd" not in design:
                raise ValueError("The only editable decision is sd (six DNA bases)")
            sd = sequence(design["sd"], 6, 6)
            leader = i["upstream_before_sd"] + sd + i["spacing_sequence"]
            artifact = {"sd": sd, "upstream_window": leader, "cds": i["fixed_cds"],
                        "dna_sequence": leader + i["fixed_cds"],
                        "model_window": leader[-35:] + i["fixed_cds"][:35]}
        else:
            if set(design) - {"utr", "rna_body", "chemistry", "cds", "polyA_tail_nt"} or "utr" not in design:
                raise ValueError("The only editable decision is utr")
            utr = sequence(design["utr"], i["variable_utr_length"], i["variable_utr_length"])
            artifact = {"utr": utr, "rna_body": (i["fixed_leader"] + utr + i["fixed_suffix"]
                        + "A" * i["polyA_tail_nt"]).replace("T", "U"),
                        "cds": i["fixed_cds"], "chemistry": i["rna_chemistry"],
                        "polyA_tail_nt": i["polyA_tail_nt"]}
        for key in design:
            if design[key] != artifact[key]:
                raise ValueError("Submitted " + key + " differs from fixed-context assembly")
        return artifact

    def evaluate(self, task, design):
        import RNA
        artifact = self.materialize(task, design)
        i = task["inputs"]
        if self.engine == "emopec_sd":
            sd, spacing = artifact["sd"], len(i["spacing_sequence"])
            value = self.emopec.get_expression(sd, spacing)
            maximum = self.emopec.get_expression(self.emopec.SEQS_MAX, spacing)
            baseline_window = (i["upstream_before_sd"] + i["baseline_sd"] + i["spacing_sequence"])[-35:] + i["fixed_cds"][:35]
            # Same 35nt+35nt window and |delta MFE| used in original make_library.
            baseline_mfe = float(RNA.fold(baseline_window.replace("T", "U"))[1])
            mfe = float(RNA.fold(artifact["model_window"].replace("T", "U"))[1])
            relative = value / maximum
            metrics = {"emopec_activity": value, "emopec_relative_activity": relative,
                       "relative_activity_target_error": abs(relative-i["target_relative_activity"]),
                       "local_mfe_kcal_mol": mfe, "baseline_local_mfe_kcal_mol": baseline_mfe,
                       "absolute_mfe_change_kcal_mol": abs(mfe-baseline_mfe),
                       "sd_edit_distance": sum(a != b for a,b in zip(sd,i["baseline_sd"]))}
            scope = ("EMOPEC empirical SD activity with original spacing correction; not absolute protein yield. "
                     "RNAfold local structural change is a computational constraint, not measured RNA stability.")
            model = {"emopec_table": "measured" if sd in self.emopec.SEQS else "predicted",
                     "ViennaRNA": RNA.__version__}
        else:
            if self.model is None:
                self.model = Optimus(self.root / self.catalog["weights_path"], self.catalog.get("model_input_length", 50))
            raw = self.model.predict([artifact["utr"]])[0]
            predicted = self.calibration["slope"] * raw + self.calibration["intercept"]
            baseline_raw = self.model.predict([i["baseline_utr"]])[0]
            baseline = self.calibration["slope"] * baseline_raw + self.calibration["intercept"]
            utr = artifact["utr"]
            changed = sum(a != b for a,b in zip(utr, i["baseline_utr"]))
            upstream = i["fixed_leader"] + utr
            aug = sum(upstream[j:j+3] == "ATG" for j in range(len(upstream)-2))
            metrics = {"optimus_standardized_score": raw, "calibrated_predicted_mrl": predicted,
                       "baseline_predicted_mrl": baseline, "predicted_mrl_change": predicted-baseline,
                       "utr_substitution_count": changed, "upstream_aug_count": aug,
                       "utr_gc_fraction": (utr.count("G")+utr.count("C"))/len(utr)}
            scope = ("Optimus5Prime original model in its fixed HEK293T/eGFP/unmodified-RNA context. "
                     "MRL scale uses a frozen independent calibration mapping; no output measurement is inferred. "
                     "No half-life endpoint is scored.")
            model = {"calibration": self.catalog["calibration_path"]}
        return {"valid": True, "metrics": metrics, "scope": scope, "model": model,
                "artifact": artifact, "experimental_outcome": "not_measured"}
