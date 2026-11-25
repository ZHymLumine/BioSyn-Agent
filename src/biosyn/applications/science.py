"""Fixed scientific profiles. Predictions and simulations are never measurements."""
import hashlib
import itertools
import json
import math
import re
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def exact_keys(value, keys):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Expected exactly these fields: {sorted(keys)}")


def sequence(value, minimum, maximum):
    if not isinstance(value, str) or not re.fullmatch(r"[ACGT]+", value):
        raise ValueError("Sequence must use uppercase DNA alphabet ACGT")
    if not minimum <= len(value) <= maximum:
        raise ValueError(f"Sequence length must be {minimum}..{maximum}")
    return value


def optimus_layers(config, input_length):
    """Read the two publisher Keras layouts, without changing layer settings."""
    entries = config["config"]
    if isinstance(entries, dict):
        if set(entries) != {"name", "layers"}:
            raise ValueError("Unsupported original Sequential metadata")
        entries = entries["layers"]
    expected = ["Conv1D", "Conv1D", "Dropout", "Conv1D", "Dropout", "Flatten",
                "Dense", "Activation", "Dropout", "Dense", "Activation"]
    if (config["class_name"] != "Sequential" or not isinstance(entries, list)
            or [x["class_name"] for x in entries] != expected):
        raise ValueError("Unsupported original Optimus architecture")
    if (input_length not in (50, 100)
            or entries[0]["config"]["batch_input_shape"] != [None, input_length, 4]):
        raise ValueError("Original model input length does not match declared encoding")
    return entries


class Optimus:
    """Explicit legacy Sequential config migration; all numerical weights unchanged."""
    def __init__(self, path, input_length=50):
        import h5py
        import tensorflow as tf
        if tf.config.threading.get_intra_op_parallelism_threads() != 2:
            tf.config.threading.set_intra_op_parallelism_threads(2)
        if tf.config.threading.get_inter_op_parallelism_threads() != 2:
            tf.config.threading.set_inter_op_parallelism_threads(2)
        with h5py.File(path) as handle:
            config = json.loads(handle.attrs["model_config"])
        entries = optimus_layers(config, input_length)
        self.input_length = input_length
        layers = [tf.keras.layers.Input(shape=(input_length, 4))]
        for entry in entries:
            settings = entry["config"].copy()
            settings.pop("batch_input_shape", None)
            layers.append(getattr(tf.keras.layers, entry["class_name"]).from_config(settings))
        self.model = tf.keras.Sequential(layers)
        self.model.load_weights(path)

    def predict(self, sequences):
        import numpy as np
        if any(set(seq)-set("ACGT") or not (25 <= len(seq) <= 100 if self.input_length == 100 else len(seq) == 50) for seq in sequences):
            raise ValueError("Sequence outside original Optimus model domain")
        padded = ["N" * (self.input_length - len(seq)) + seq for seq in sequences]
        encoded = np.array([[[int(base == nuc) for nuc in "ACGT"] for base in seq]
                            for seq in padded], dtype=np.float32)
        return self.model(encoded, training=False).numpy().reshape(-1).tolist()


class Evaluator:
    def __init__(self, profile_path, resource_root):
        self.profile_path = Path(profile_path)
        self.profile = json.loads(self.profile_path.read_text())
        self.root = Path(resource_root).resolve()
        self.profile_sha256 = digest(profile_path)
        for item in self.profile["resources"]:
            path = (self.root / item["path"]).resolve()
            if not path.is_relative_to(self.root) or digest(path) != item["sha256"]:
                raise ValueError(f"Resource hash mismatch: {item['path']}")
        self.model = None
        self.catalog = None

    def evaluate(self, design):
        kind = self.profile["kind"]
        result = getattr(self, kind)(design)
        result.update(profile_sha256=self.profile_sha256,
                      evidence_type=self.profile["evidence_type"],
                      limitations=self.profile["limitations"])
        # Reject nonfinite numbers even if a numerical backend returned them.
        json.dumps(result, allow_nan=False)
        return result

    def bacterial(self, design):
        from ostir import run_ostir
        exact_keys(design, ["utr"])
        utr = sequence(design["utr"], *self.profile["utr_length"])
        full = utr + self.profile["fixed_cds_window"]
        start = len(utr) + 1
        rows = run_ostir(full, start=start, end=start, threads=1, aSD="ACCUCCUUA", decimal_places=8)
        if len(rows) != 1 or rows[0]["start_position"] != start:
            raise RuntimeError("OSTIR did not return exactly the annotated initiation site")
        row = rows[0]
        tir = float(row["expression"])
        if tir <= 0:
            raise RuntimeError("Nonpositive OSTIR initiation rate")
        err = abs(math.log2(tir / self.profile["target_tir"]))
        return {"valid": True, "feasible": err <= self.profile["max_abs_log2_error"],
                "metrics": {"predicted_tir": tir, "abs_log2_target_error": err,
                            "utr_length": len(utr)}, "raw": row,
                "artifact": {"utr": utr, "fixed_cds_window": self.profile["fixed_cds_window"],
                             "start_position_1based": start}}

    def mammalian(self, design):
        exact_keys(design, ["utr"])
        utr = sequence(design["utr"], 50, 50)
        if self.model is None:
            self.model = Optimus(self.root / self.profile["weights"])
        score = self.model.predict([utr])[0]
        gc = (utr.count("G") + utr.count("C")) / 50
        uaug = sum(utr[i:i+3] == "ATG" for i in range(48))
        violations = []
        if not self.profile["gc_range"][0] <= gc <= self.profile["gc_range"][1]:
            violations.append("GC fraction outside public design constraint")
        if uaug:
            violations.append("Upstream AUG present")
        if score < self.profile["minimum_scaled_score"]:
            violations.append("Predicted standardized MRL below requested threshold")
        return {"valid": True, "feasible": not violations,
                "metrics": {"optimus_scaled_mrl": score, "gc_fraction": gc, "upstream_aug": uaug},
                "violations": violations, "artifact": {"utr": utr, "cds_context": "original eGFP assay"}}

    def cello_catalog(self):
        if self.catalog is not None:
            return self.catalog
        ucf = json.loads((self.root / self.profile["ucf"]).read_text())
        inputs = json.loads((self.root / self.profile["inputs"]).read_text())
        funcs = {x["name"]: x for x in ucf if x["collection"] == "functions"}
        if funcs["Hill_response"]["equation"] != "ymin + (ymax - ymin) / (1.0 + (x / K)^n)":
            raise ValueError("Unsupported gate response equation")
        if funcs["linear_input_composition"]["equation"] != "x2 + x1":
            raise ValueError("Unsupported input composition")
        models = {x["name"]: x for x in ucf + inputs if x["collection"] == "models"}
        gates = {}
        for gate in (x for x in ucf if x["collection"] == "gates"):
            model = models[gate["model"]]
            if (model["functions"]["response_function"] != "Hill_response" or
                    model["functions"]["input_composition"] != "linear_input_composition"):
                raise ValueError("Mixed unsupported gate model")
            gates[gate["name"]] = {"group": gate["group"], "regulator": gate["regulator"],
                "parameters": {p["name"]: p["value"] for p in model["parameters"]}}
        sensors = {s["name"]: {p["name"]: p["value"] for p in models[s["model"]]["parameters"]}
                   for s in inputs if s["collection"] == "input_sensors"}
        self.catalog = {"gates": gates, "sensors": sensors,
                        "host": next(x for x in ucf if x["collection"] == "header")}
        return self.catalog

    def circuit(self, design, sensor_scale=1.0):
        exact_keys(design, ["gates", "output"])
        gates = design["gates"]
        if not isinstance(gates, list) or not 1 <= len(gates) <= self.profile["max_gates"]:
            raise ValueError("Circuit gate count outside profile")
        catalog = self.cello_catalog()
        seen = set(self.profile["sensors"])
        groups = set()
        for gate in gates:
            exact_keys(gate, ["id", "part", "inputs"])
            if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,39}", gate["id"]) or gate["id"] in seen:
                raise ValueError("Gate ids must be unique and separate from sensor names")
            part = catalog["gates"][gate["part"]]
            if part["group"] in groups:
                raise ValueError("Repeated regulator family is not orthogonal")
            groups.add(part["group"])
            if (not isinstance(gate["inputs"], list) or not 1 <= len(gate["inputs"]) <= 2 or
                    len(set(gate["inputs"])) != len(gate["inputs"]) or not set(gate["inputs"]) <= seen):
                raise ValueError("One or two unique, already defined inputs required (topological DAG order)")
            seen.add(gate["id"])
        if design["output"] not in {g["id"] for g in gates}:
            raise ValueError("Output must name a gate")
        rows = []
        for bits in itertools.product((0, 1), repeat=2):
            values, logic = {}, dict(zip(self.profile["sensors"], bits))
            for sensor, bit in logic.items():
                params = catalog["sensors"][sensor]
                values[sensor] = params["ymax" if bit else "ymin"] * sensor_scale
            for gate in gates:
                p = catalog["gates"][gate["part"]]["parameters"]
                x = sum(values[i] for i in gate["inputs"])
                values[gate["id"]] = p["ymin"] + (p["ymax"]-p["ymin"])/(1+(x/p["K"])**p["n"])
                logic[gate["id"]] = int(not any(logic[i] for i in gate["inputs"]))
            target = int(all(bits))
            rows.append({"inputs": list(bits), "target": target, "logic": logic[design["output"]],
                         "output_rpu": values[design["output"]], "gate_rpu": values})
        on = min(row["output_rpu"] for row in rows if row["target"])
        off = max(row["output_rpu"] for row in rows if not row["target"])
        correct = all(row["logic"] == row["target"] for row in rows)
        ratio = on / off
        return {"valid": True, "feasible": correct and ratio >= self.profile["minimum_on_off_ratio"],
                "metrics": {"on_off_ratio": ratio, "minimum_on_rpu": on, "maximum_off_rpu": off,
                            "logic_correct": correct, "gate_count": len(gates)},
                "truth_table": rows, "artifact": design}

    def metabolic_model(self):
        if self.model is None:
            from cobra.io import read_sbml_model
            self.model = read_sbml_model(self.root / self.profile["model"])
            self.model.solver = "glpk"
            # Explicit medium replaces source medium; no silent boundary fallbacks.
            self.model.medium = self.profile["medium"]
        return self.model

    def metabolic(self, design, glucose_scale=1.0):
        exact_keys(design, ["gene_knockouts"])
        genes = design["gene_knockouts"]
        if (not isinstance(genes, list) or len(genes) > self.profile["max_knockouts"] or
                any(not isinstance(g, str) for g in genes) or len(set(genes)) != len(genes)):
            raise ValueError("Expected unique gene ids within the knockout budget")
        model = self.metabolic_model()
        for gene in genes:
            if gene not in model.genes:
                raise ValueError(f"Unknown iJO1366 gene {gene}")
        with model:
            model.reactions.EX_glc__D_e.lower_bound *= glucose_scale
            original_bounds = {r.id: r.bounds for r in model.reactions}
            for gene in genes:
                model.genes.get_by_id(gene).knock_out()
            knockout_effects = [{"reaction": r.id, "gene_rule": r.gene_reaction_rule,
                                 "before": list(original_bounds[r.id]), "after": list(r.bounds)}
                                for r in model.reactions if r.bounds != original_bounds[r.id]]
            growth = model.reactions.get_by_id(self.profile["biomass"])
            product = model.reactions.get_by_id(self.profile["product_exchange"])
            model.objective = growth
            sol = model.optimize()
            if sol.status != "optimal":
                return {"valid": True, "feasible": False, "metrics": {"knockout_count": len(genes)},
                        "solver_status": sol.status, "knockout_effects": knockout_effects,
                        "violations": ["No feasible steady-state growth solution"]}
            maximum_growth = sol.objective_value
            if maximum_growth < self.profile["minimum_growth"]:
                return {"valid": True, "feasible": False,
                        "metrics": {"growth_max_per_hour": maximum_growth, "knockout_count": len(genes)},
                        "knockout_effects": knockout_effects,
                        "violations": ["Growth below public minimum"]}
            growth.lower_bound = self.profile["minimum_growth"]
            model.objective = product
            model.objective_direction = "max"
            prodsol = model.optimize()
            if prodsol.status != "optimal":
                raise RuntimeError(f"Production optimization status: {prodsol.status}")
            maximum_product = prodsol.objective_value
            # Lower and upper product bounds near the design's OWN maximum growth.
            growth.lower_bound = max(self.profile["minimum_growth"], 0.99 * maximum_growth)
            extrema = {}
            for direction in ("min", "max"):
                model.objective_direction = direction
                s = model.optimize()
                if s.status != "optimal":
                    raise RuntimeError(f"Product FVA failed: {s.status}")
                extrema[direction] = s.objective_value
                if direction == "min":
                    minimum_solution = s
            pathways = []
            for rxn in model.reactions:
                flux = float(prodsol.fluxes[rxn.id])
                if abs(flux) > 1e-7 and any(m.id in self.profile["pathway_metabolites"] for m in rxn.metabolites):
                    pathways.append({"reaction": rxn.id, "equation": rxn.reaction, "flux": flux,
                                     "gene_rule": rxn.gene_reaction_rule,
                                     "mass_charge_imbalance": rxn.check_mass_balance()})
            return {"valid": True, "feasible": extrema["min"] >= self.profile["minimum_coupled_flux"],
                    "metrics": {"growth_max_per_hour": maximum_growth,
                        "succinate_max_at_growth_floor": maximum_product,
                        "succinate_min_at_99pct_growth": extrema["min"],
                        "succinate_max_at_99pct_growth": extrema["max"], "knockout_count": len(genes)},
                    "artifact": {"gene_knockouts": genes, "active_product_pathway": pathways},
                    "knockout_effects": knockout_effects,
                    "minimum_product_solution": [{"reaction": r.id, "name": r.name,
                        "gene_rule": r.gene_reaction_rule, "flux": float(minimum_solution.fluxes[r.id])}
                        for r in model.reactions if r.id in
                        {"ACKr", "PTAr", "ALCD2x", "ACALD", "LDH_D", "PFL", "GLCptspp",
                         "PYK", "PPC", "SUCDi", "FRD2", "FRD3"} or
                        (r.id.startswith("EX_") and abs(float(minimum_solution.fluxes[r.id])) > 1e-7)],
                    "solver_status": "optimal"}

    def describe(self):
        if self.profile["kind"] == "circuit":
            return self.cello_catalog()
        if self.profile["kind"] == "metabolic":
            model = self.metabolic_model()
            return {"model_id": model.id, "genes": [{"id": g.id, "name": g.name} for g in model.genes],
                    "reactions": [{"id": r.id, "name": r.name, "equation": r.reaction,
                                   "gene_rule": r.gene_reaction_rule} for r in model.reactions],
                    "medium": model.medium,
                    "note": "Only gene knockouts are editable; no exchange, objective or stoichiometry edits."}
        return {"baseline": self.profile["baseline"], "context": self.profile["context"],
                "limitations": self.profile["limitations"]}
