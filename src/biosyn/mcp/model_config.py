"""Deployment metadata for isolated foundation-model workers."""
import json
import os
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import Literal

from pydantic import Field

from biosyn.agent.state import StrictModel


class ModelConfig(StrictModel):
    backend: Literal["evo2", "nt", "nt_head", "rnafm"]
    model_id: str
    revision: str = Field(min_length=1, description="Checkpoint version or immutable revision")
    checkpoint: str
    device: str = "cpu"
    max_length: int = Field(default=1000, ge=1)
    trust_remote_code: bool = False
    prediction_task: str | None = None
    unit: str | None = None


@lru_cache(maxsize=1)
def model_config():
    path = os.environ.get("BIOSYN_MODEL_CONFIG")
    if not path:
        raise ValueError("BIOSYN_MODEL_CONFIG must identify an explicit model deployment JSON")
    raw = Path(path).read_bytes()
    config = ModelConfig.model_validate(json.loads(raw))
    checkpoint = Path(config.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Configured checkpoint is missing: {checkpoint}")
    metadata = {"model_id": config.model_id, "revision": config.revision,
                "checkpoint": str(checkpoint.resolve()), "device": config.device,
                "deployment_manifest_sha256": sha256(raw).hexdigest()}
    return config, metadata
