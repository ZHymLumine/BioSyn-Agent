"""Cello UCF-backed NOR circuits: one graph determines both sequence and response."""

import json
import math
import itertools
from pathlib import Path


def validate_truth_table(truth_table, input_names=None):
    """Refuse partial scoring of unsupported multi-output tasks before running any model."""
    if not truth_table:
        raise ValueError("Empty truth table")
    names = list(input_names or truth_table[0]["inputs"])
    if any(set(r["outputs"]) != {"output_YFP"} for r in truth_table):
        raise ValueError("This Cello evaluator supports only output_YFP; multi-output tasks require all reporter resources")
    states = []
    for row in truth_table:
        if set(row["inputs"]) != set(names) or any(v not in (0, 1) for v in row["inputs"].values()):
            raise ValueError("Truth-table inputs must be complete Boolean states")
        if row["outputs"]["output_YFP"] not in (0, 1):
            raise ValueError("Truth-table outputs must be Boolean")
        states.append(tuple(row["inputs"][n] for n in names))
    if len(set(states)) != len(states) or set(states) != set(itertools.product((0, 1), repeat=len(names))):
        raise ValueError("Truth table must cover every input state exactly once")
    if {r["outputs"]["output_YFP"] for r in truth_table} != {0, 1}:
        raise ValueError("ON/OFF evaluation requires both ON and OFF states")


class CircuitEvaluator:
    def __init__(self, catalog_path):
        data = json.loads(Path(catalog_path).read_text())
        records = sum(data["sections"].values(), [])
        self.parts = {r["name"]: r["dnasequence"].upper() for r in records if r["collection"] == "parts"}
        self.structures = {r["name"]: r for r in records if r["collection"] == "structures"}
        self.models = {r["name"]: r for r in records if r["collection"] == "models"}
        self.gates = {r["name"]: r for r in records if r["collection"] == "gates"}
        functions = {r["name"]: r for r in records if r["collection"] == "functions"}
        if functions["Hill_response"]["equation"] != "ymin + (ymax - ymin) / (1.0 + (x / K)^n)":
            raise ValueError("Unsupported UCF Hill equation")
        if functions["linear_input_composition"]["equation"] != "x2 + x1":
            raise ValueError("Unsupported input composition")
        self.sensors = {}
        for r in records:
            if r["collection"] == "input_sensors":
                promoter = self.structures[r["structure"]]["outputs"][0]
                self.sensors["input_" + promoter] = {
                    "promoter": promoter,
                    "parameters": self.parameters(r["model"]),
                }

    def parameters(self, model):
        return {p["name"]: p["value"] for p in self.models[model]["parameters"]}

    def describe(self):
        return {
            "gates": {
                k: {"group": v["group"], "parameters": self.parameters(v["model"])}
                for k, v in self.gates.items()
            },
            "sensors": self.sensors,
            "model_scope": "Eco1C1G1T1 steady-state Hill response, additive tandem promoters; no dynamics.",
        }

    def assemble(self, gates, output, input_names):
        """Assemble exactly the submitted NOR graph using original device templates."""
        promoters = {name: self.sensors[name]["promoter"] for name in input_names}
        nodes = [{"id": name, "name": name, "type": "INPUT", "inputs": []} for name in input_names]
        families, assembly = set(), []
        for gate in gates:
            if set(gate) != {"id", "part", "inputs"}:
                raise ValueError("A physical gate requires exactly id, part and inputs")
            gid, sources = gate["id"], gate["inputs"]
            if not isinstance(gid, str) or gid in promoters or gid == "output_YFP":
                raise ValueError("Duplicate or reserved gate id")
            if (
                not isinstance(sources, list)
                or not 1 <= len(sources) <= 2
                or len(set(sources)) != len(sources)
            ):
                raise ValueError("Each NOR gate takes one or two distinct promoters")
            if any(s not in promoters for s in sources):
                raise ValueError("Unknown input or graph not in topological order")
            entry = self.gates[gate["part"]]
            if entry["group"] in families:
                raise ValueError("Regulator families must be orthogonal within a design")
            families.add(entry["group"])
            structure = self.structures[entry["structure"]]
            devices = {d["name"]: d["components"] for d in structure["devices"]}

            def expand(component):
                if component.startswith("#in"):
                    index = int(component[3:]) - 1
                    return [promoters[sources[index]]] if index < len(sources) else []
                if component in devices:
                    return [p for child in devices[component] for p in expand(child)]
                if component not in self.parts:
                    raise ValueError("Unknown physical part " + component)
                return [component]

            names = expand(gate["part"])
            assembly.append({"node_id": gid, "part_names": names})
            promoters[gid] = structure["outputs"][0]
            nodes.append(
                {
                    "id": gid,
                    "name": gate["part"],
                    "type": "NOT" if len(sources) == 1 else "NOR",
                    "inputs": sources,
                }
            )
        if output not in promoters:
            raise ValueError("Output does not name a physical signal")
        assembly.append({"node_id": "output_YFP", "part_names": [promoters[output], "YFP_cassette"]})
        nodes.append({"id": "output_YFP", "name": "output_YFP", "type": "OUTPUT", "inputs": [output]})
        dna = "".join(self.parts[p] for unit in assembly for p in unit["part_names"])
        return {"gates": gates, "output": output, "nodes": nodes, "assembly": assembly, "dna_sequence": dna}

    def evaluate(self, task, design, sensor_scale=1.0):
        validate_truth_table(task["inputs"]["truth_table"], task["inputs"]["input_signals"])
        expected = self.assemble(design["gates"], design["output"], list(task["inputs"]["input_signals"]))
        for key in ["nodes", "assembly", "dna_sequence"]:
            if design.get(key) != expected[key]:
                raise ValueError(f"{key} does not match the submitted physical graph and UCF templates")
        rows = []
        for truth in task["inputs"]["truth_table"]:
            logic = dict(truth["inputs"])
            values = {
                k: self.sensors[k]["parameters"]["ymax" if v else "ymin"] * sensor_scale
                for k, v in logic.items()
            }
            for gate in design["gates"]:
                entry = self.gates[gate["part"]]
                p = self.parameters(entry["model"])
                x = sum(values[s] for s in gate["inputs"])
                values[gate["id"]] = p["ymin"] + (p["ymax"] - p["ymin"]) / (1 + (x / p["K"]) ** p["n"])
                logic[gate["id"]] = int(not any(logic[s] for s in gate["inputs"]))
            output = design["output"]
            rows.append(
                {
                    "inputs": truth["inputs"],
                    "target": truth["outputs"]["output_YFP"],
                    "boolean_output": logic[output],
                    "output_promoter_rpu": values[output],
                }
            )
        on = min(r["output_promoter_rpu"] for r in rows if r["target"])
        off = max(r["output_promoter_rpu"] for r in rows if not r["target"])
        accuracy = sum(r["boolean_output"] == r["target"] for r in rows) / len(rows)
        return {
            "valid": True,
            "metrics": {
                "truth_table_accuracy": accuracy,
                "on_off_ratio": on / off,
                "log10_on_off_ratio": math.log10(on / off),
                "minimum_on_rpu": on,
                "maximum_off_rpu": off,
                "gate_count": len(design["gates"]),
                "dna_length": len(design["dna_sequence"]),
            },
            "model_function_satisfied": accuracy == 1 and on > off,
            "truth_table": rows,
            "scope": self.describe()["model_scope"],
            "sequence_matches_physical_graph": True,
        }


def synthesize_nor(truth_table):
    """Generic Boolean synthesis from the public truth table; no reference circuit access."""
    from sympy import symbols
    from sympy.logic.boolalg import SOPform, POSform, And, Or, Not

    validate_truth_table(truth_table)
    names = list(truth_table[0]["inputs"])
    symbols_ = symbols(" ".join(names))
    ones = [[r["inputs"][n] for n in names] for r in truth_table if r["outputs"]["output_YFP"]]
    expressions = [SOPform(symbols_, ones), POSform(symbols_, ones)]

    def build(expression):
        gates, cache = [], {}

        def nor(xs):
            key = tuple(sorted(set(xs)))
            if key not in cache:
                gid = f"g{len(gates) + 1}"
                gates.append({"id": gid, "inputs": list(key)})
                cache[key] = gid
            return cache[key]

        def invert(x):
            existing = next((g for g in gates if g["id"] == x), None)
            if existing is not None and len(existing["inputs"]) == 1:
                return existing["inputs"][0]
            return nor([x])

        def convert(expr):
            if expr in symbols_:
                return str(expr)
            if expr.func == Not:
                return invert(convert(expr.args[0]))
            values = [convert(e) for e in expr.args]
            if expr.func == And:
                result = nor([invert(values[0]), invert(values[1])])
                for v in values[2:]:
                    result = nor([invert(result), invert(v)])
                return result
            if expr.func == Or:
                result = invert(nor(values[:2]))
                for v in values[2:]:
                    result = invert(nor([result, v]))
                return result
            raise ValueError("Constant or unsupported Boolean function")

        output = convert(expression)
        # Prune unused internal signals produced by algebraic simplification.
        used = {output}
        for gate in reversed(gates):
            if gate["id"] in used:
                used.update(gate["inputs"])
        return {
            "gates": [g for g in gates if g["id"] in used],
            "output": output,
            "expression": str(expression),
        }

    alternatives = [build(expr) for expr in expressions]
    return min(alternatives, key=lambda item: (len(item["gates"]), item["expression"]))
