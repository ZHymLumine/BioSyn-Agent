import math
from typing import Any

from Bio.Seq import Seq
from Bio.Data.CodonTable import TranslationError

from biosyn.agent.optimizer import score
from biosyn.agent.state import Candidate, Evaluation, TaskSpec


def at_path(value: Any, path: str):
    for part in path.split("."):
        value = value[int(part)] if isinstance(value, list) else value[part]
    return value


def evaluate(candidate: Candidate, task: TaskSpec, gateway) -> Evaluation:
    result = Evaluation(candidate_id=candidate.id)
    if task.required_protein is not None and candidate.protein != task.required_protein:
        result.violations.append("Protein sequence differs from required_protein")
    if task.require_transcription_match:
        if not candidate.dna or not candidate.rna or candidate.dna.replace("T", "U") != candidate.rna:
            result.violations.append("RNA must be the transcription of candidate DNA")
    if task.require_translation_match:
        if not candidate.dna or not candidate.protein:
            result.violations.append("Translation check requires DNA and protein")
        else:
            try:
                translated = str(Seq(candidate.dna).translate(table=task.translation_table, cds=True))
                if translated != candidate.protein:
                    result.violations.append("Protein differs from complete CDS translation")
            except (ValueError, TranslationError) as exc:
                result.violations.append(f"Invalid complete CDS: {exc}")
    for spec in task.evaluators:
        try:
            arguments = dict(spec.arguments)
            for arg, field in spec.inputs.items():
                value = at_path(candidate.model_dump(), field)
                if value is None:
                    raise ValueError(f"Candidate field {field} is missing")
                arguments[arg] = value
            receipt = gateway.call(spec.tool, arguments)
            result.call_ids.append(receipt["call_id"])
            data = receipt["data"]
            for name, path in spec.metrics.items():
                value = at_path(data, path)
                if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                    raise ValueError(f"{name} must be a finite numeric tool result")
                result.metrics[name] = float(value)
        except Exception as exc:
            # Evaluation is a failure boundary: preserve diagnostics and prohibit acceptance.
            result.errors.append(f"{spec.name}: {type(exc).__name__}: {exc}")
    if not result.errors:
        result.score, violations = score(result.metrics, task.objectives)
        result.violations.extend(violations)
    result.passed = not result.errors and not result.violations
    return result
