# OpenBioMed workflows

The OpenBioMed adapter exposes named workflows as MCP tools. It runs in the OpenBioMed environment and converts typed protein, molecule, text and JSON inputs into workflow inputs.

## Configure workflows

Set up [OpenBioMed](https://github.com/PharMolix/OpenBioMed) and its workflow dependencies, then install BioSyn in that environment. The adapter uses `open_biomed.core.workflow.Workflow`; the source interface corresponds to OpenBioMed revision `5ddc32de1849b3e8c41dd69c71761afc27899480`.

Create a JSON file mapping workflow names to YAML files:

```json
{
  "my_workflow": "/path/to/OpenBioMed/workflows/my_workflow.yaml"
}
```

Set the interpreter and mapping paths:

```bash
export BIOSYN_OPENBIOMED_PYTHON=/path/to/openbiomed-env/bin/python
export BIOSYN_OPENBIOMED_WORKFLOWS=/path/to/workflows.local.json
```

Add this server to the runtime configuration:

```yaml
servers:
  openbiomed:
    command: ["${BIOSYN_OPENBIOMED_PYTHON}", "-m", "biosyn.integrations.openbiomed"]
    env:
      BIOSYN_OPENBIOMED_WORKFLOWS: "${BIOSYN_OPENBIOMED_WORKFLOWS}"
    timeout: 1800
```

## Execute a workflow

`openbiomed.list_workflows` lists the configured names and files. `openbiomed.run_workflow` accepts a workflow name, typed inputs and a repeat count:

```json
{
  "name": "my_workflow",
  "inputs": [
    {"node": 0, "field": "protein", "kind": "protein", "value": "MAKELV"}
  ],
  "repeats": 1
}
```

Choose node indices and field names from the workflow YAML. Available input kinds are `json`, `protein`, `molecule` and `text`. Protein inputs use amino-acid sequences; molecule inputs use SMILES.

Sequence, molecule and text outputs are serialized into MCP results. Workflows producing structural protein objects should include an artifact-export node that returns a file path.
