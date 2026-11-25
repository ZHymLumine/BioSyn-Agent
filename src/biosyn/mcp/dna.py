"""Lightweight DNA tools; foundation models live in separate servers/environments."""
from importlib.metadata import version

from Bio.Seq import Seq
from mcp.server.fastmcp import FastMCP

from biosyn.mcp.common import digest, sequence, serve

server = FastMCP("BioSyn DNA", log_level="WARNING")


@server.tool()
def sequence_analyze(dna: str) -> dict:
    """Count canonical DNA bases and GC fraction. These are sequence statistics, not expression predictions."""
    dna = sequence(dna, "ACGT")
    return {"length": len(dna), "gc_fraction": (dna.count("G") + dna.count("C")) / len(dna),
            "sha256": digest(dna), "evidence_type": "computed"}


@server.tool()
def transcribe(dna: str) -> dict:
    """Transcribe the supplied coding-strand DNA by T-to-U replacement."""
    dna = sequence(dna, "ACGT")
    return {"rna": str(Seq(dna).transcribe()), "input_sha256": digest(dna), "evidence_type": "computed"}


@server.tool()
def translate_cds(dna: str, table: int = 1) -> dict:
    """Translate a complete CDS, requiring a valid start, stop, and reading frame in the specified genetic code."""
    dna = sequence(dna, "ACGT")
    protein = str(Seq(dna).translate(table=table, cds=True))
    return {"protein": protein, "length": len(protein), "table": table,
            "input_sha256": digest(dna), "backend": f"biopython/{version('biopython')}",
            "evidence_type": "computed"}


if __name__ == "__main__":
    serve(server)
