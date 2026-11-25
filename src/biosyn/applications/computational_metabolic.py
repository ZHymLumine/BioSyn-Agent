"""Curated reaction designs evaluated in a declared genome-scale host model."""

import json
from pathlib import Path


def routes_to_product(task, resources):
    target = task["inputs"]["product_metabolite"]
    starts = task["inputs"]["central_precursors"]
    graph = {}
    for rid, entry in resources["reactions"].items():
        graph.setdefault(entry["main_substrate"], []).append((rid, entry["main_product"]))
    routes = []

    def walk(node, path, seen):
        if node == target:
            routes.append(path)
            return
        if len(path) >= 8:
            return
        for rid, end in sorted(graph.get(node, [])):
            if end not in seen:
                walk(end, path + [rid], seen | {end})

    for start in starts:
        walk(start, [], {start})
    if not routes:
        raise ValueError(
            "No route connects the declared central precursors to the target in the public library"
        )
    return routes


def initial_pathway(task, resources, rng):
    return {"steps": rng.choice(routes_to_product(task, resources)), "gene_knockouts": []}


def mutate_pathway(design, task, resources, rng, index):
    if index % 3 == 0:
        design["steps"] = rng.choice(routes_to_product(task, resources))
    else:
        choices = resources["allowed_knockouts"]
        current = set(design["gene_knockouts"])
        if current and (len(current) >= task["inputs"]["max_knockouts"] or rng.random() < 0.4):
            current.remove(rng.choice(sorted(current)))
        else:
            available = sorted(set(choices) - current)
            if available:
                current.add(rng.choice(available))
        design["gene_knockouts"] = sorted(current)
    return design


class MetabolicEvaluator:
    def __init__(self, task, catalog_path, resource_root):
        from cobra.io import read_sbml_model

        self.catalog = json.loads(Path(catalog_path).read_text())
        conditions = task["inputs"]["computational_conditions"]
        self.model = read_sbml_model(Path(resource_root) / conditions["model_path"])
        self.model.solver = "glpk"
        if self.model.id != conditions["model_id"]:
            raise ValueError("Host model identity differs from the public conditions")
        self.model.medium = conditions["medium"]
        glucose = self.model.reactions.get_by_id(conditions["substrate_exchange"])
        glucose.bounds = (-conditions["substrate_uptake"], -conditions["substrate_uptake"])
        self.growth = self.model.reactions.get_by_id(conditions["biomass_reaction"])
        self.model.objective = self.growth
        wild = self.model.optimize()
        if wild.status != "optimal" or wild.objective_value <= 0:
            raise ValueError("Declared host medium does not support wild-type growth")
        self.wild_growth = float(wild.objective_value)
        self.floor = self.wild_growth * conditions["minimum_growth_fraction_of_wildtype"]
        self.conditions = conditions
        # Do not permit unreported carbon feed through another boundary.
        carbon_inputs = []
        for r in self.model.exchanges:
            if r.lower_bound < 0 and any(m.elements.get("C", 0) > 0 for m in r.metabolites):
                carbon_inputs.append(r.id)
        if carbon_inputs != [conditions["substrate_exchange"]]:
            raise ValueError("Unreported carbon source in declared medium: " + str(carbon_inputs))

    def _check_design(self, task, design):
        if set(design) != {"steps", "gene_knockouts"}:
            raise ValueError("Metabolic decisions require exactly steps and gene_knockouts")
        steps = design["steps"]
        genes = design["gene_knockouts"]
        if not isinstance(steps, list) or not steps or len(steps) > 8 or len(set(steps)) != len(steps):
            raise ValueError("Expected 1..8 unique reaction/enzyme option IDs")
        if (
            not isinstance(genes, list)
            or len(set(genes)) != len(genes)
            or len(genes) > task["inputs"]["max_knockouts"]
        ):
            raise ValueError("Invalid gene knockout list")
        if set(genes) - set(self.catalog["allowed_knockouts"]):
            raise ValueError("Knockout not in the public allowed gene list")
        current = None
        for rid in steps:
            entry = self.catalog["reactions"][rid]
            if current is None:
                if entry["main_substrate"] not in task["inputs"]["central_precursors"]:
                    raise ValueError("Route must begin at a declared central-metabolism precursor")
            elif current != entry["main_substrate"]:
                raise ValueError("Declared reactions do not form an ordered connected route")
            current = entry["main_product"]
        if current != task["inputs"]["product_metabolite"]:
            raise ValueError("Route does not reach the specified product")
        return steps, genes

    def evaluate(self, task, design):
        from cobra import Metabolite, Reaction
        from cobra.flux_analysis import pfba

        steps, genes = self._check_design(task, design)
        m = self.model
        with m:
            for mid, spec in self.catalog["new_metabolites"].items():
                if mid not in m.metabolites:
                    m.add_metabolites(
                        [
                            Metabolite(
                                mid,
                                name=spec["name"],
                                formula=spec["formula"],
                                charge=spec["charge"],
                                compartment="c",
                            )
                        ]
                    )
            selected = []
            artifacts = []
            hetero = set()
            native = set()
            for rid in steps:
                item = self.catalog["reactions"][rid]
                if item["operation"] in ("native_reference", "enable_reverse"):
                    r = m.reactions.get_by_id(item["model_reaction"])
                    direction = item["model_direction"]
                    before = list(r.bounds)
                    if item["operation"] == "enable_reverse":
                        # This is an explicit selected engineering operation, never an implicit FBA repair.
                        r.lower_bound = min(r.lower_bound, -1000)
                    observed = {met.id: float(v) * direction for met, v in r.metabolites.items()}
                    if observed != item["stoichiometry"]:
                        raise ValueError("Native reaction does not match the curated directed equation")
                    native.update(item["genes"])
                elif item["operation"] == "add_enzyme":
                    r = Reaction("BS_" + rid, lower_bound=0, upper_bound=1000)
                    r.add_metabolites(
                        {m.metabolites.get_by_id(mid): v for mid, v in item["stoichiometry"].items()}
                    )
                    engineered = item.get("engineered_gene_ids", item["uniprot"])
                    if not engineered or len(set(engineered)) != len(engineered):
                        raise ValueError("An added enzyme needs explicit unique engineered gene identities")
                    r.gene_reaction_rule = " and ".join("bs_" + x for x in engineered)
                    m.add_reactions([r])
                    direction = 1
                    before = None
                    hetero.update(engineered)
                else:
                    raise ValueError("Unsupported curated reaction operation")
                imbalance = r.check_mass_balance()
                if imbalance:
                    raise ValueError("Unbalanced curated reaction " + rid + ": " + str(imbalance))
                selected.append((rid, r, direction))
                artifacts.append(
                    {
                        "option_id": rid,
                        "reaction": r.reaction,
                        "direction": direction,
                        "genes": item["genes"],
                        "organism": item["organism"],
                        "uniprot": item["uniprot"],
                        "engineered_gene_ids": item.get("engineered_gene_ids", item["uniprot"]),
                        "catalytic_support": item.get("catalytic_support", []),
                        "identity_scope": item.get("identity_scope", "original catalog annotation"),
                        "sources": item["sources"],
                        "operation": item["operation"],
                        "bounds_before": before,
                        "bounds_after": list(r.bounds),
                        "mass_charge_imbalance": imbalance,
                    }
                )
            for gene in genes:
                m.genes.get_by_id(gene).knock_out()
            target = m.metabolites.get_by_id(task["inputs"]["product_metabolite"])
            export = Reaction("BS_PRODUCT_EXPORT", lower_bound=0, upper_bound=1000)
            export.add_metabolites({target: -1})
            m.add_reactions([export])
            # Every proposed branch step must carry at least the product flux. All current library
            # steps produce one unit of their named main product per forward reaction event.
            for index, (rid, r, direction) in enumerate(selected):
                m.add_cons_vars(
                    m.problem.Constraint(
                        direction * r.flux_expression - export.flux_expression,
                        lb=0,
                        name=f"proposed_branch_{index}",
                    )
                )
            metrics = {
                "pathway_step_count": len(steps),
                "heterologous_gene_count": len(hetero),
                "native_pathway_gene_count": len(native),
                "knockout_count": len(genes),
                "wildtype_growth_per_hour": self.wild_growth,
                "minimum_growth_per_hour": self.floor,
                "product_yield_mol_per_mol": 0.0,
                "carbon_yield_fraction": 0.0,
                "product_flux": 0.0,
            }
            m.objective = self.growth
            growth_solution = m.optimize()
            if growth_solution.status != "optimal" or growth_solution.objective_value < self.floor - 1e-8:
                return {
                    "valid": True,
                    "feasible": False,
                    "metrics": metrics,
                    "violations": ["Required growth is infeasible"],
                    "pathway": artifacts,
                    "scope": self.scope(),
                }
            maxgrowth = float(growth_solution.objective_value)
            metrics["maximum_growth_per_hour"] = maxgrowth
            self.growth.lower_bound = self.floor
            m.objective = export
            solution = m.optimize()
            if solution.status != "optimal":
                raise RuntimeError("Product optimization failed: " + solution.status)
            optimum = float(solution.objective_value)
            # Parsimonious flux solution after maximizing the product, reducing arbitrary flux loops.
            fluxes = pfba(m, fraction_of_optimum=0.999999).fluxes
            carbon_yield = optimum * target.elements["C"] / (self.conditions["substrate_uptake"] * 6)
            if carbon_yield > 1 + 1e-6:
                raise ValueError("Carbon yield exceeds glucose carbon input")
            metrics.update(
                product_flux=optimum,
                product_yield_mol_per_mol=optimum / self.conditions["substrate_uptake"],
                carbon_yield_fraction=carbon_yield,
                biomass_flux_at_product_optimum=float(fluxes[self.growth.id]),
            )
            for row, (_, r, direction) in zip(artifacts, selected):
                row["directed_flux_at_product_optimum"] = float(fluxes[r.id]) * direction
            with m:
                self.growth.lower_bound = 0
                m.reactions.get_by_id(self.conditions["substrate_exchange"]).bounds = (0, 0)
                m.reactions.ATPM.lower_bound = 0
                zero = m.optimize()
                if zero.status != "optimal" or zero.objective_value > 1e-6:
                    raise ValueError("Failed zero-carbon production check")
                metrics["zero_carbon_product_flux"] = float(zero.objective_value)
            with m:
                self.growth.lower_bound = max(self.floor, 0.99 * maxgrowth)
                extrema = {}
                for direction in ["min", "max"]:
                    m.objective_direction = direction
                    sol = m.optimize()
                    if sol.status != "optimal":
                        raise RuntimeError("Growth-constrained product range failed: " + sol.status)
                    extrema[direction] = float(sol.objective_value)
                metrics["product_min_at_99pct_growth"] = extrema["min"]
                metrics["product_max_at_99pct_growth"] = extrema["max"]
            exchanges = {r.id: float(fluxes[r.id]) for r in m.exchanges if abs(float(fluxes[r.id])) > 1e-7}
            return {
                "valid": True,
                "feasible": optimum > 1e-6,
                "metrics": metrics,
                "pathway": artifacts,
                "exchange_fluxes_at_product_optimum": exchanges,
                "scope": self.scope(),
                "experimental_outcome": "not_measured",
            }

    def scope(self):
        return (
            "iJO1366 K-12 steady-state stoichiometric upper bound in the explicitly declared medium. "
            "Product export is represented by a free cytosolic sink; transport, toxicity, enzyme kinetics "
            "and expression burden are not predicted. Gene count is an engineering-complexity proxy. "
            "Selected branch reactions are coupled to export to assess the proposed route, not an unspecified alternative."
        )
