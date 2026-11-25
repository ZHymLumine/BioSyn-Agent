"""Nucleotide Transformer MCP worker; no implicit downloads or random prediction head."""
import contextlib
import sys
import threading
from functools import lru_cache

from mcp.server.fastmcp import FastMCP

from biosyn.mcp.common import digest, sequence, serve
from biosyn.mcp.model_config import model_config

server = FastMCP("BioSyn Nucleotide Transformer", log_level="WARNING")
_lock = threading.Lock()


@lru_cache(maxsize=1)
def load_model():
    config, _ = model_config()
    if config.backend not in {"nt", "nt_head"}:
        raise ValueError("Nucleotide Transformer service requires backend=nt or nt_head")
    from transformers import AutoConfig, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoTokenizer
    kwargs = dict(local_files_only=True, trust_remote_code=config.trust_remote_code)
    with contextlib.redirect_stdout(sys.stderr):
        tokenizer = AutoTokenizer.from_pretrained(config.checkpoint, **kwargs)
        model_config_hf = AutoConfig.from_pretrained(config.checkpoint, **kwargs)
        cls = AutoModelForMaskedLM
        if config.backend == "nt_head":
            if not config.prediction_task or not config.unit:
                raise ValueError("A trained head needs prediction_task and unit metadata")
            if not any("ForSequenceClassification" in name for name in (model_config_hf.architectures or [])):
                raise ValueError("Checkpoint does not declare a trained sequence prediction head")
            cls = AutoModelForSequenceClassification
        model, loading = cls.from_pretrained(config.checkpoint, output_loading_info=True, **kwargs)
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise ValueError(f"Incomplete checkpoint loading: {loading}")
        model = model.to(config.device).eval()
    return tokenizer, model


def encode(dna):
    config, _ = model_config()
    sequence(dna, "ACGT", config.max_length)
    tokenizer, model = load_model()
    encoded = tokenizer(dna, return_tensors="pt", truncation=False)
    limit = min(tokenizer.model_max_length, getattr(model.config, "max_position_embeddings", tokenizer.model_max_length))
    if encoded["input_ids"].shape[1] > limit:
        raise ValueError("Tokenized input exceeds model context; truncation is not allowed")
    # NT's ESM tokenizer can report an EOS mask entry without appending an EOS
    # token. Derive exclusions from the actual ids, so the mask stays aligned.
    special = encoded["input_ids"].eq(tokenizer.all_special_ids[0])
    for special_id in tokenizer.all_special_ids[1:]:
        special |= encoded["input_ids"].eq(special_id)
    special = special.to(config.device)
    return {k: v.to(config.device) for k, v in encoded.items()}, special, model


@server.tool()
def nt_embed(dna: str) -> dict:
    """Return mean Nucleotide Transformer token embeddings excluding special tokens; no phenotype is predicted."""
    import torch
    config, metadata = model_config()
    with _lock, contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
        encoded, special, model = encode(dna)
        output = model(**encoded, output_hidden_states=True)
        mask = encoded["attention_mask"].bool() & ~special.bool()
        vector = output.hidden_states[-1][0, mask[0]].float().mean(0)
        if not torch.isfinite(vector).all():
            raise ValueError("Nonfinite embedding")
    return {"embedding": vector.cpu().tolist(), "dimension": vector.numel(),
            "input_sha256": digest(dna), "evidence_type": "representation", "model": metadata}


@server.tool()
def nt_predict(dna: str) -> dict:
    """Apply an explicitly configured, trained sequence prediction head; refuses a base masked language model."""
    import torch
    config, metadata = model_config()
    if config.backend != "nt_head":
        raise ValueError("nt_predict requires a trained nt_head checkpoint; embeddings are not predictions")
    with _lock, contextlib.redirect_stdout(sys.stderr), torch.inference_mode():
        encoded, _, model = encode(dna)
        logits = model(**encoded).logits[0].float()
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite prediction")
    return {"logits": logits.cpu().tolist(), "labels": model.config.id2label,
            "prediction_task": config.prediction_task, "unit": config.unit,
            "input_sha256": digest(dna), "evidence_type": "predicted", "model": metadata}


if __name__ == "__main__":
    model_config()
    serve(server)
