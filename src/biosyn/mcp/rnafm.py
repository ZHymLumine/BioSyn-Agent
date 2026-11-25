"""RNA-FM representation service in its own model environment."""
import contextlib
import sys
import threading
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from biosyn.mcp.common import digest, sequence, serve
from biosyn.mcp.model_config import model_config

server = FastMCP("BioSyn RNA-FM", log_level="WARNING")
_lock = threading.Lock()


@lru_cache(maxsize=1)
def load_model():
    config, _ = model_config()
    if config.backend != "rnafm":
        raise ValueError("RNA-FM service requires backend=rnafm")
    import fm
    with contextlib.redirect_stdout(sys.stderr):
        model, alphabet = fm.pretrained.rna_fm_t12(model_location=config.checkpoint)
        model = model.to(config.device).eval()
    return model, alphabet.get_batch_converter()


@server.tool()
def rnafm_embed(rna: str) -> dict:
    """Return the mean layer-12 RNA-FM residue representation; no translation efficiency or degradation score."""
    import torch
    config, metadata = model_config()
    rna = sequence(rna, "ACGU", min(config.max_length, 1022))
    with _lock, contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
        model, converter = load_model()
        _, _, tokens = converter([("candidate", rna)])
        output = model(tokens.to(config.device), repr_layers=[12])
        vector = output["representations"][12][0, 1:len(rna) + 1].float().mean(0)
        if not torch.isfinite(vector).all():
            raise ValueError("Nonfinite RNA-FM representation")
    return {"embedding": vector.cpu().tolist(), "dimension": vector.numel(), "layer": 12,
            "input_sha256": digest(rna), "evidence_type": "representation", "model": metadata}


if __name__ == "__main__":
    model_config()
    serve(server)
