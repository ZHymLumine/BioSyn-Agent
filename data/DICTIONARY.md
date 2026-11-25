# Data dictionary

## Case index

`cases.tsv` contains one row per design task. `dataset` names the task version;
`case_id` is the stable case identifier; `track` is the runtime track;
`family_id` groups related source contexts; `evaluation_engine` selects the
scientific evaluator. Paths are relative to `data/`. A blank index field means
that the original manifest has no value for that field.

Runtime track names `bacterial_cds` and `mammalian_mrna` are retained for API
compatibility. Their formal tasks edit the SD and 5′UTR, respectively.

## Task JSON

| Field | Meaning |
|---|---|
| `case_id`, `track` | Case and application identifiers |
| `request` | Natural-language design problem |
| `inputs` | Host, fixed sequences, starting design, editable region and conditions |
| `expected_output` | Required biological design fields |
| `evaluation_contract.requirements` | Metric, operator, target and optional numerical tolerance |
| `evaluation_contract.threshold_basis` | How the task's acceptance targets were chosen |
| `inputs.public_resources` | Files exposed to the agent, with SHA-256 bindings |

Sequences use uppercase A/C/G/T or A/C/G/U as specified by the field. Sequence
lengths and edit counts are integers in nucleotides. JSON `null` represents an
unspecified value. Numbers in evaluation contracts are acceptance targets.

## Profile JSON

`decision_schema` describes the editable design fields. `hard_constraints` fixes
the biological context. `objectives` lists ranking metrics and directions.
`budget` records tool, token, candidate and episode limits; `replicates` records
algorithm seeds. These fields control execution and evaluation.

## Reference JSON

`design` holds the reference decision; `artifact` holds an assembled design where
available; `evaluation` records its computed properties. Additional fields
identify the source case, original table row or accession, selection procedure,
historical anchor and feasible alternatives. References are curator/evaluator
data and stay outside the agent's mounted task directory.

Published measurements in `source_case.anchor` or `feasibility_witness` refer to
those exact source sequences and assays. Predictor outputs and reference-route
fluxes are labeled in their evaluation records.

## Metrics

| Metric | Unit or scale |
|---|---|
| `truth_table_accuracy` | Fraction of input states implemented correctly |
| `on_off_ratio` | Minimum ON response / maximum OFF response |
| `gate_count` | Physical gate count |
| `emopec_relative_activity` | Activity normalized to the EMOPEC maximum at the fixed spacing |
| `relative_activity_target_error` | Absolute difference on the normalized activity scale |
| `absolute_mfe_change_kcal_mol` | Absolute local RNAfold MFE difference, kcal/mol |
| `calibrated_predicted_mrl` | Predicted mean ribosome load after affine calibration |
| `predicted_mrl_change` | Designed minus baseline predicted MRL |
| `utr_substitution_count` | Hamming distance from the baseline UTR |
| `product_yield_mol_per_mol` | mol product/mol glucose |
| `pathway_step_count` | Number of selected branch reactions |
| `heterologous_gene_count` | Introduced catalytic genes and required support roles |

The exact operators and numerical tolerances are preserved in each task and
collected in [objectives.json](objectives.json).

## Manifests and provenance

The top-level [manifest.json](manifest.json) records each packaged file's path,
size and SHA-256. [provenance/file-origins.json](provenance/file-origins.json)
maps copied files to their original server-relative paths. Scientific resource
locks contain file hashes and package versions. Historical paths in those locks
describe the original deployment; `prepare_dataset.py` creates explicit bindings
for a new checkout.
