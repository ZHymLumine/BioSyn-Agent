# Application design and evaluation

Application episodes extend the sequence `TaskSpec` interface with a task-specific design schema, public resources and a host-side scientific evaluator.

The repository includes the [22-case formal dataset](../data/README.md), with
task inputs, reference designs, source metadata and evaluator calibration. Follow
its preparation commands to obtain scientific assets and create a dataset copy
bound to your checkout.

## Task and candidate interfaces

An application task contains `case_id`, `track`, `request`, `inputs`, `decision_schema` and `evaluation_contract`. A submitted candidate contains `id`, `parent_id`, `design` and `rationale`. The design fields depend on the task:

| Track | Design fields | Evaluator |
|---|---|---|
| `genetic_circuit` | Regulatory `gates` and `output` | `CircuitEvaluator` |
| `bacterial_cds` with `emopec_sd` | `sd` | `RegulatoryEvaluator` |
| `mammalian_mrna` with `optimus_utr` | `utr` | `RegulatoryEvaluator` |
| `metabolic_pathway` | `steps`, `gene_knockouts` | `MetabolicEvaluator` |

`evaluation_contract.requirements` declares the metric, comparison operator and target for each design objective. Scientific services assemble the design and compute its metrics. The host ledger records submissions and supplies feedback to the planner.

The application service exposes `evaluate_design` and track-specific assembly tools. Circuit tasks also expose NOR-topology synthesis; pathway tasks expose connected-route enumeration.

## Scientific resources

Scientific resources are supplied through a resource directory and catalogues:

- **Circuits:** a Cello gate and part catalogue, input devices, output devices and response parameters.
- **Bacterial regulation:** EMOPEC software and sequence context, with ViennaRNA for local folding energy.
- **Human regulation:** Optimus5Prime weights, input length and calibration parameters for the transcript context.
- **Pathways:** a COBRA-compatible SBML host model, reaction catalogue and medium/growth conditions.
- **Additional evaluation:** OSTIR, RiboNN checkpoints, Saluki checkpoints, Cello cytometry distributions and a second host reconstruction.

Resource locks map root-relative filenames to SHA-256 values and package versions. `verify_resources` checks these bindings when the scientific service starts.

The model adapters use the upstream scientific software in its corresponding environment. Install COBRApy for metabolic evaluation; TensorFlow and h5py for Optimus5Prime and Saluki; and PyTorch plus the RiboNN source environment for RiboNN.

## Run a scientific MCP service

With a task, catalogue and resource lock prepared:

```bash
python -m biosyn.mcp.computational \
  --root /path/to/project \
  --task /path/to/task.json \
  --catalog /path/to/catalog.json \
  --resources /path/to/scientific-resources \
  --resource-lock /path/to/resource-lock.json
```

This starts a stdio MCP service. It can be registered under `servers.science` with these arguments.

## Application episode runner

`biosyn.applications.comparison` runs complete design episodes from a dataset manifest. Its Linux deployment uses Bubblewrap (`bwrap`), `.venv-biomni/` for the planner, `.venv-design-eval/` for scientific evaluation, and `resources/application_suite_v3/` for model resources. The endpoint file contains the LLM `base_url` and an `env_file` defining `BIOSYN_GLM_API_KEY`; the runner's configured model is `zai-org/GLM-5.3-Flash`.

A dataset directory provides:

```text
dataset/
├── manifest.json         # cases and source-file bindings
├── public/               # requests and public design resources
├── profiles/             # design objectives and episode budgets
└── resource-locks/       # scientific software and resource bindings
```

Each manifest case identifies its `task_path`, `profile_path`, `resources_path`, `catalog_path` and `resource_lock_path`. The manifest's `files` list provides the paths and SHA-256 values, and its `status` is `ready` when the resources are prepared.

```bash
python -m biosyn.applications.comparison \
  --root . \
  --dataset /path/to/dataset \
  --endpoint /path/to/endpoint.local.json \
  --output runs/application-design \
  --methods biosyn \
  --seeds 101 \
  --workers 1
```

The runner supports `llm_direct`, `biomni_upstream`, `biosyn_no_goal_feedback` and `biosyn`, along with additional comparison modes. `--resume` continues an existing schedule using the same code and task configuration.

## Additional evaluation

`biosyn.applications.external_evaluation.ExternalEvaluationSuite` evaluates completed designs with additional sequence predictors, a second host reconstruction or gate-response perturbations. It keeps each output in its scientific units and records which model and resource produced it.

Application data and model checkpoints are configured separately from the framework package. The [demo](https://zhymlumine.github.io/biosyn-agent-demo/) shows the resulting design and evaluation workflow.
