import os
import sys
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, model_validator

from biosyn.agent.state import StrictModel


class ServerConfig(StrictModel):
    transport: Literal["stdio", "streamable-http"] = "stdio"
    command: list[str] = Field(default_factory=list)
    url: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    headers_env: dict[str, str] = Field(default_factory=dict)
    cwd: str | None = None
    enabled: bool = True
    timeout: float = Field(default=120, gt=0, le=86400)

    @model_validator(mode="after")
    def transport_fields(self):
        if self.enabled:
            if self.transport == "stdio" and not self.command:
                raise ValueError("stdio server requires a command")
            if self.transport == "streamable-http" and not self.url:
                raise ValueError("HTTP server requires a URL")
        return self


class LLMConfig(StrictModel):
    model: str
    base_url: str
    api_key_env: str = "BIOSYN_LLM_API_KEY"
    temperature: float = 0.2
    max_tokens: int = Field(default=4096, gt=0)
    timeout: float | None = Field(default=180, gt=0)
    extra_body: dict = Field(default_factory=dict)
    api_socket: str | None = None


class RuntimeConfig(StrictModel):
    servers: dict[str, ServerConfig]
    llm: LLMConfig | None = None
    skills: list[str] = Field(default_factory=list)
    biomni_tool_modules: list[str] | None = None
    biomni_excluded_tools: list[str] = Field(default_factory=list)
    preserve_native_biomni_knowhow: bool = False
    register_runtime_tools: bool = True
    use_tool_retriever: bool = True
    recursion_limit: int = Field(default=40, ge=2, le=500)
    code_timeout: int = Field(default=180, ge=1)

    @model_validator(mode="after")
    def server_names(self):
        import re
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n) for n in self.servers):
            raise ValueError("Server names must be Python identifiers without dots")
        return self


def expand(value: str) -> str:
    if value == "{python}":
        return sys.executable
    if value.startswith("${") and value.endswith("}"):
        name = value[2:-1]
        if not os.environ.get(name):
            raise ValueError(f"Required environment variable {name} is unset")
        return os.environ[name]
    return value


def load_config(path: Path) -> RuntimeConfig:
    return RuntimeConfig.model_validate(yaml.safe_load(path.read_text()))
