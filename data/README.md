# BioSyn-Agent design tasks

This dataset contains **22 natural-language biological design tasks** across four
applications. Each case specifies a biological context, editable design choices,
an output format and quantitative computational objectives. The task inputs,
reference designs and evaluation profiles preserve the server dataset version
`formal-v2-expanded-20261102`.

| Application | Cases | Design output | Evaluation |
|---|---:|---|---|
| Genetic circuits | 6 | Physical NOR-gate connections and regulatory DNA | Boolean function, Cello ON/OFF ratio and gate count |
| Bacterial RBS | 5 | Six-base SD sequence in a fixed E. coli gene context | EMOPEC activity-target error and local RNAfold energy change |
| Human 5′UTR | 6 | Variable UTR in a fixed HEK293T/eGFP transcript | Calibrated Optimus mean ribosome load, edit count and predicted change |
| Metabolic pathways | 5 | Ordered reaction/enzyme choices and support genes | iJO1366 FBA yield, pathway length and gene count |

Browse [cases.tsv](cases.tsv) for the complete case index and
[objectives.json](objectives.json) for the exact acceptance criteria. The task
requests retain the Chinese text used in the experiments. Field names, identifiers
and this guide are in English.

## Files

```text
data/
├── formal_v2_expanded_20261102/
│   ├── public/tasks/       Natural-language requests and design constraints
│   ├── public/resources/   Agent-visible component and reaction information
│   ├── profiles/           Decision schemas, objectives and execution budgets
│   ├── private/            Reference designs and their calibration evaluations
│   ├── resource-locks/     Original evaluator file and package bindings
│   └── manifest.json       Original case and file index
├── calibration/            Two fixed Optimus-to-MRL calibration mappings
├── provenance/             Original-to-release file mapping
├── cases.tsv               Case index
├── objectives.json         Case-specific goal requirements
├── sources.json            Publications, accessions and source versions
├── external-resources.json Download URLs and SHA-256 values
└── manifest.json           Packaged file inventory
```

The `private/` name describes access during an agent episode: these reference files
are included for readers and evaluator development. The agent receives its task,
profile and public resources. References and calibration records remain on the
evaluation host. Reference evaluations document how the tasks and targets were
constructed.

## Load and check the data

Reading and validating the task files uses the Python standard library:

```python
import json
from pathlib import Path

dataset = Path("data/formal_v2_expanded_20261102")
manifest = json.loads((dataset / "manifest.json").read_text())
case = manifest["cases"][0]
task = json.loads((dataset / case["task_path"]).read_text())
print(task["case_id"], task["request"])
print(task["evaluation_contract"]["requirements"])
```

```bash
python scripts/check_data.py
```

The first check verifies bundled files and identifies the separately downloaded
Cello catalogue. Obtain its original UCF, input and output records and reconstruct
the exact catalogue used by the tasks:

```bash
python scripts/fetch_data.py --group catalog
python scripts/check_data.py --require-catalog
```

Every download and the reconstructed catalogue are checked against their recorded
SHA-256 values. Downloaded upstream files remain in the local workspace and are
excluded from Git.

## Prepare a design run

Fetch the original EMOPEC implementation, two Optimus checkpoints and the iJO1366
model, and install the bundled calibration mappings:

```bash
python scripts/fetch_data.py --group evaluation
python scripts/prepare_dataset.py --output runs/prepared-formal-data
```

Preparation creates a new dataset copy with resource paths bound to the current
checkout. Task requests, profiles, reference designs, model hashes and scientific
thresholds stay unchanged. `preparation.json` records the original and prepared
bindings. The original resource locks remain in `data/` as provenance.

Install the track-specific package versions recorded in `resource-locks/` in the
scientific evaluation environment. The application runner uses Linux, Bubblewrap,
`.venv-biomni/`, `.venv-design-eval/` and an LLM endpoint, as described in the
[application guide](../docs/applications.md). For example:

```bash
python -m biosyn.applications.comparison \
  --root . \
  --dataset runs/prepared-formal-data \
  --endpoint /path/to/endpoint.local.json \
  --output runs/design-evaluation \
  --methods biosyn \
  --seeds 101 \
  --workers 1
```

The profiles record the original run budgets. The current runner uses the episode
deadline for LLM calls, so the historical per-request timeout fields are retained
as provenance. Seeds 101, 202 and 303 identify repeated algorithm runs.

## Task construction and measurement units

- **Circuits:** six single-output truth tables are paired with the Eco1C1G1T1
  component library. A design must implement the truth table, separate ON and OFF
  states and respect its gate budget. Response values use relative promoter units;
  ON/OFF ratio and truth-table accuracy are dimensionless.
- **Bacterial RBS:** five native gene contexts from E. coli accession U00096.3
  provide a fixed CDS, upstream window and spacing. Only six SD bases are editable.
  Targets use normalized EMOPEC activity; local folding-energy change is measured
  in kcal/mol over the specified 35 + 35 nucleotide window.
- **Human 5′UTR:** four 50-nucleotide tasks tune translation upward or downward
  within an edit budget; two tasks design UTRs of 25 and 100 nucleotides. Contexts
  and historical anchors come from Sample et al. The supplied affine mappings
  place model outputs on the mean-ribosome-load scale. MRL is dimensionless.
- **Pathways:** five products use curated reaction and enzyme choices under a
  fixed glucose-only iJO1366 medium and growth floor. Yield is mol product/mol
  glucose; flux is mmol/gDW/h; growth is h⁻¹. Gene count measures engineering
  complexity. Yield thresholds are based on feasible reference routes under each
  case's step and gene limits.

The complete schemas and variable definitions are in the
[data dictionary](DICTIONARY.md). The reference records distinguish published
observations, source sequences, constructed reference designs and computed
quantities.

## Sources and reuse

[SOURCES.md](SOURCES.md) links each application to its original publications,
repositories and accessions. [sources.json](sources.json) provides the same
information in machine-readable form. The original UTR tables and circuit
reference files can be retrieved with:

```bash
python scripts/fetch_data.py --group source-tables
```

BioSyn-authored task specifications and documentation follow the repository's
[Apache-2.0 license](../LICENSE). Reused sequences, records, scientific software
and model assets retain their upstream attribution and terms. The download
manifest identifies the exact upstream assets; the code license applies to the
BioSyn-authored portions.
