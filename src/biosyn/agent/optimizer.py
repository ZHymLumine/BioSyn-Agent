from biosyn.agent.state import Evaluation, Objective


def score(metrics: dict[str, float], objectives: list[Objective]) -> tuple[float, list[str]]:
    violations = []
    total = 0.0
    for objective in objectives:
        value = metrics[objective.metric]  # Missing evidence must fail, never become zero.
        if objective.minimum is not None and value < objective.minimum:
            violations.append(f"{objective.metric}={value:g} < {objective.minimum:g}")
        if objective.maximum is not None and value > objective.maximum:
            violations.append(f"{objective.metric}={value:g} > {objective.maximum:g}")
        utility = min(1.0, max(0.0, (value - objective.lower) / (objective.upper - objective.lower)))
        if objective.direction == "minimize":
            utility = 1.0 - utility
        total += objective.weight * utility
    return total / sum(o.weight for o in objectives), violations


def select(evaluations: dict[str, Evaluation], objectives: list[Objective]):
    feasible = [e for e in evaluations.values() if e.passed and e.score is not None]
    best = max(feasible, key=lambda e: e.score).candidate_id if feasible else None

    def dominates(a, b):
        differences = [(a.metrics[o.metric] - b.metrics[o.metric]) *
                       (1 if o.direction == "maximize" else -1) for o in objectives]
        return all(d >= 0 for d in differences) and any(d > 0 for d in differences)

    frontier = [e.candidate_id for e in feasible if not any(dominates(other, e) for other in feasible)]
    return best, frontier
