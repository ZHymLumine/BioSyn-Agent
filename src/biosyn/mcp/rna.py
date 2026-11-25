from importlib.metadata import version

import RNA
from mcp.server.fastmcp import FastMCP

from biosyn.mcp.common import digest, sequence, serve

server = FastMCP("BioSyn RNA", log_level="WARNING")


@server.tool()
def rnafold(rna: str, temperature_celsius: float = 37.0) -> dict:
    """Predict RNA MFE secondary structure with ViennaRNA. MFE is not measured stability or translation efficiency."""
    rna = sequence(rna, "ACGU", max_length=2000)
    if not 0 <= temperature_celsius <= 100:
        raise ValueError("temperature_celsius must be between 0 and 100")
    model = RNA.md()
    model.temperature = temperature_celsius
    structure, energy = RNA.fold_compound(rna, model).mfe()
    return {"structure": structure, "mfe_kcal_mol": energy, "length": len(rna),
            "gc_fraction": (rna.count("G") + rna.count("C")) / len(rna),
            "temperature_celsius": temperature_celsius, "input_sha256": digest(rna),
            "evidence_type": "predicted", "backend": f"ViennaRNA/{version('ViennaRNA')}"}


if __name__ == "__main__":
    serve(server)
