"""Evo2 MCP service. Install Evo2 in its own CUDA environment before enabling."""
import contextlib
import math
import sys
import threading
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from biosyn.mcp.common import digest, sequence, serve
from biosyn.mcp.model_config import model_config

server = FastMCP("BioSyn Evo2", log_level="WARNING")
_lock = threading.Lock()


@lru_cache(maxsize=1)
def load_model():
    config, _ = model_config()
    if config.backend != "evo2":
        raise ValueError("This service requires backend=evo2")
    from evo2 import Evo2
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Evo2 requires an allocated CUDA GPU")
    with contextlib.redirect_stdout(sys.stderr):
        return Evo2(config.model_id, local_path=config.checkpoint)


@server.tool()
def evo2_score(dna: str) -> dict:
    """Return mean sequence log-likelihood from Evo2; it is not expression, fitness, or functional activity."""
    config, metadata = model_config()
    dna = sequence(dna, "ACGT", config.max_length)
    with _lock, contextlib.redirect_stdout(sys.stderr):
        value = float(load_model().score_sequences([dna], batch_size=1, reduce_method="mean")[0])
    if not math.isfinite(value):
        raise ValueError("Evo2 produced a nonfinite likelihood")
    return {"mean_log_likelihood": value, "input_sha256": digest(dna),
            "evidence_type": "model_likelihood", "model": metadata}


@server.tool()
def evo2_generate(prompt: str, n_tokens: int = 64, temperature: float = 1.0, top_k: int = 4, seed: int = 0) -> dict:
    """Generate a DNA continuation with the configured Evo2 checkpoint and explicit random seed."""
    config, metadata = model_config()
    prompt = sequence(prompt, "ACGT", config.max_length)
    if not 1 <= n_tokens <= config.max_length - len(prompt):
        raise ValueError("Prompt plus generated tokens exceeds deployment length budget")
    if not math.isfinite(temperature) or temperature <= 0 or not 1 <= top_k <= 512:
        raise ValueError("Invalid sampling parameters")
    import torch
    with _lock, contextlib.redirect_stdout(sys.stderr):
        model = load_model()
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        output = model.generate(prompt_seqs=[prompt], n_tokens=n_tokens,
                                temperature=temperature, top_k=top_k, verbose=0)
        generated = output.sequences[0]
    sequence(generated, "ACGT", config.max_length)
    return {"sequence": generated, "prompt": prompt, "seed": seed, "n_tokens": n_tokens,
            "temperature": temperature, "top_k": top_k, "evidence_type": "generated", "model": metadata}


if __name__ == "__main__":
    model_config()
    serve(server)
