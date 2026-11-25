# MCP tools and model services

BioSyn-Agent keeps persistent MCP sessions with local stdio workers or Streamable HTTP services. Tools are registered with Biomni from the schemas advertised by each server.

## Local sequence tools

The CPU configuration launches these services using the active Python interpreter:

```yaml
servers:
  dna:
    command: ["{python}", "-m", "biosyn.mcp.dna"]
  rna:
    command: ["{python}", "-m", "biosyn.mcp.rna"]
```

The tool catalogue includes `dna.sequence_analyze`, `dna.transcribe`, `dna.translate_cds` and `rna.rnafold`. During planning, qualified names are exposed as Python functions such as `dna__transcribe`.

## HTTP services

Start an RNA service:

```bash
python -m biosyn.mcp.rna --transport streamable-http --host 127.0.0.1 --port 8787
```

Connect to it from a configuration:

```yaml
servers:
  rna:
    transport: streamable-http
    url: http://127.0.0.1:8787/mcp
    timeout: 120
```

For an authenticated service, `headers_env` maps an HTTP header name to an environment variable containing its value.

## Foundation models

Each foundation-model worker runs in an environment containing its backend and the BioSyn package. Install the backend using its upstream instructions, then install BioSyn into that environment:

```bash
python -m pip install -e /path/to/BioSyn-Agent
```

| Worker module | Backend | Operations |
|---|---|---|
| `biosyn.mcp.evo2` | [Evo2](https://github.com/ArcInstitute/evo2) | `evo2_generate`, `evo2_score` |
| `biosyn.mcp.nt` | [Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer) | `nt_embed`, `nt_predict` |
| `biosyn.mcp.rnafm` | [RNA-FM](https://github.com/ml4bio/RNA-FM) | `rnafm_embed` |

A deployment manifest specifies the model and local checkpoint:

```json
{
  "backend": "nt",
  "model_id": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
  "revision": "REPLACE_WITH_CHECKPOINT_REVISION",
  "checkpoint": "models/nucleotide-transformer",
  "device": "cpu",
  "max_length": 1000,
  "trust_remote_code": true
}
```

Save the manifest as a local configuration and point `BIOSYN_MODEL_CONFIG` to it. The `nt_head` backend additionally takes `prediction_task` and `unit` for a trained sequence-classification checkpoint. Evo2 uses a CUDA GPU; NT and RNA-FM use the device declared in their manifests.

Copy the server entries from [`extensions.example.yaml`](../configs/biomni/extensions.example.yaml) into your runtime configuration. Set the worker's interpreter and manifest paths through the corresponding environment variables, then set `enabled: true`.

The embedding services return sequence representations. Evo2 scoring returns mean sequence log-likelihood. Task objectives can use a trained prediction head or another scientific evaluator with the required output quantity.

## Protein tools

Install the optional protein dependency:

```bash
python -m pip install -e '.[protein]'
biosyn catalog \
  --config configs/biomni/protein-cpu.yaml \
  --run-dir runs/protein-catalog
```

The configuration starts the upstream `protein_design_mcp.server`. Configure its model dependencies and device according to the [protein-design-mcp documentation](https://github.com/jasonkim8652/protein-design-mcp).

## Add a tool

An MCP server can expose a typed scientific function:

```python
from mcp.server.fastmcp import FastMCP

server = FastMCP("Sequence properties")

@server.tool()
def sequence_length(sequence: str) -> dict:
    """Return sequence length in residues or nucleotides."""
    return {"length": len(sequence)}

if __name__ == "__main__":
    server.run()
```

Register its command under `servers`, inspect it with `biosyn catalog`, and map its output to an objective using an `EvaluationSpec`. See the [task guide](tasks.md).
