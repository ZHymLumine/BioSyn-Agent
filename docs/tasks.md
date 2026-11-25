# Defining a design task

A `TaskSpec` combines a natural-language request with objectives, evaluators and budgets. The planner proposes `Candidate` objects; the runtime evaluates them and maintains the candidate history.

## Candidate fields

| Field | Meaning |
|---|---|
| `id` | Unique candidate identifier |
| `parent_id` | Identifier of the candidate being revised |
| `dna`, `rna`, `protein` | Uppercase sequences in their canonical alphabets |
| `artifacts` | Named paths to additional design files |
| `phenotype` | Requested or annotated properties |
| `rationale` | Explanation of the proposed design |

For a CDS task, `require_transcription_match` checks the DNA–RNA relationship and `require_translation_match` checks translation under the selected genetic code. `required_protein` fixes the amino-acid sequence.

## Objectives and evaluator mappings

This objective constrains GC content while ranking accepted candidates by increasing GC fraction:

```json
{
  "metric": "gc_fraction",
  "direction": "maximize",
  "lower": 0,
  "upper": 1,
  "minimum": 0.40,
  "maximum": 0.65,
  "weight": 1
}
```

`minimum` and `maximum` define acceptance bounds. `lower` and `upper` set the scale for the weighted score. The runtime also records Pareto-optimal accepted candidates.

The evaluator maps a candidate field to an MCP argument and extracts a numeric result:

```json
{
  "name": "dna_statistics",
  "tool": "dna.sequence_analyze",
  "inputs": {"dna": "dna"},
  "arguments": {},
  "metrics": {"gc_fraction": "gc_fraction"}
}
```

`inputs` maps tool argument names to candidate fields. `arguments` supplies constant tool arguments. `metrics` maps objective names to dotted paths in the returned data. Each objective has one evaluator.

The complete example is [`examples/tasks/cds_design.json`](../examples/tasks/cds_design.json).

## Evaluate supplied candidates

Add candidates to the task's `candidates` list and run:

```bash
biosyn evaluate \
  --config configs/biomni/cpu.yaml \
  --task examples/tasks/evaluate_cds.json \
  --run-dir runs/cds-evaluation
```

This executes the configured MCP evaluators directly. The task in this example contains an input sequence, and evaluation outputs are generated in the run directory.

## Budgets and execution

- `max_rounds`: planner rounds.
- `max_candidates`: submitted candidate count.
- `max_tool_calls`: calls through the scientific tool gateway.

Model output length, request timeout, graph recursion and code execution time are configured separately in the runtime YAML. `llm.timeout: null` keeps a request open until the model server responds; `code_timeout` bounds each executed code block.

The planner uses Biomni's Python/Bash execution environment. Run it in a dedicated environment with the files and tools needed for the task. Application episodes use the Linux sandbox described in the [application guide](applications.md).

## Reading a run

The runtime creates a new run directory and updates `state.json` after candidate submissions and evaluations. `report.json` contains the final state. A successful run also writes `best-design.json` and `best-design.fasta`.

A candidate is accepted when its sequence checks and all objective bounds pass. Tool errors and constraint violations are returned to the planner as feedback for the next revision.
