import json

import pytest

from biosyn.config import expand
from biosyn.mcp.model_config import ModelConfig, model_config
from biosyn.integrations.openbiomed import workflow_paths


def test_required_environment_does_not_silently_expand_to_empty(monkeypatch):
    monkeypatch.delenv("BIOSYN_MISSING_TEST_VARIABLE", raising=False)
    with pytest.raises(ValueError, match="unset"):
        expand("${BIOSYN_MISSING_TEST_VARIABLE}")


def test_foundation_checkpoint_must_exist(tmp_path, monkeypatch):
    manifest = tmp_path / "model.json"
    manifest.write_text(ModelConfig(backend="nt", model_id="test", revision="test-revision",
                                   checkpoint=str(tmp_path / "missing")).model_dump_json())
    monkeypatch.setenv("BIOSYN_MODEL_CONFIG", str(manifest))
    model_config.cache_clear()
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        model_config()
    model_config.cache_clear()


def test_openbiomed_workflow_discovery_does_not_import_heavy_models(tmp_path, monkeypatch):
    manifest = tmp_path / "workflows.json"
    manifest.write_text(json.dumps({"example": "/not-installed/example.yaml"}))
    monkeypatch.setenv("BIOSYN_OPENBIOMED_WORKFLOWS", str(manifest))
    assert workflow_paths() == {"example": "/not-installed/example.yaml"}
