"""Actual Biomni A1 loop; scientific evaluation/ledger remain outside its namespace."""
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
from pydantic import BaseModel

from biosyn.agent.planner import BiomniPlanner
from biosyn.agent.state import RunStore
from biosyn.config import RuntimeConfig


class Task(BaseModel):
    id: str
    request: str


class Episode:
    def __init__(self):
        self.task = json.loads(Path("/public/task.json").read_text())
        self.store = RunStore(Path("/work/output"))
        self.client = httpx.Client(transport=httpx.HTTPTransport(uds="/proxy/eval.sock"),
                                   base_url="http://evaluation", timeout=240)
        self.gateway = SimpleNamespace(catalog={})
        self.state = SimpleNamespace(run_id=self.task["id"], task=Task(id=self.task["id"],
            request=self.task["request"] + " Read /public/profile.json for baseline, constraints and metric meanings. "
                "Read /public/catalog.json selectively for original gate parameters or model gene/reaction ids. "
                "All authoritative scientific feedback comes from biosyn_submit. Files under /public are read-only."))

    def rpc(self, action, payload):
        response = self.client.post("/" + action, json=payload)
        if response.status_code != 200:
            raise ValueError(response.text)
        return response.json()

    def summary(self):
        return self.rpc("state", {})

    def submit(self, candidate):
        return self.rpc("submit", {"candidate": candidate})

    def finalize(self, candidate_id):
        return self.rpc("finalize", {"candidate_id": candidate_id})

    def run(self):
        config = RuntimeConfig.model_validate({"servers": {}, "biomni_tool_modules": [],
            "use_tool_retriever": True, "recursion_limit": 80, "code_timeout": 180,
            "llm": {"model": "deepseek-v4-flash", "base_url": "http://deepseek.local/v1",
                "api_socket": "/proxy/llm.sock", "api_key_env": "BIOSYN_PROXY_KEY", "temperature": 0.2,
                "max_tokens": 4096, "timeout": 240, "extra_body": {"thinking": {"type": "disabled"}}}})
        schema = {"type": "object", "required": ["id", "parent_id", "design", "rationale"],
            "properties": {"id": {"type": "string"}, "parent_id": {"type": ["string", "null"]},
                "design": {"type": "object"}, "rationale": {"type": "string"}}, "additionalProperties": False}
        planner = BiomniPlanner(config, self, submission_schema=schema,
            instructions=("You are BioSyn-Agent using Biomni A1 for a real computational design pilot. "
                "First read the profile and submit its baseline with id baseline, parent_id null. "
                "Inspect the actual returned feedback before deciding how to revise. Submit changed designs "
                "with unique ids, parent_id naming a previously evaluated design, and rationale explaining the feedback used. "
                "Do not fabricate or calculate replacement scores. biosyn_submit executes the fixed scientific MCP evaluator "
                "and returns predictions/simulations. You have 24 candidate evaluations and 32 LLM requests. "
                "Use only a small fraction initially, reflect on returned results, then refine. "
                "Read biosyn_get_state to monitor remaining LLM requests as well as candidate evaluations. "
                "Reserve the last three LLM requests for final selection and call biosyn_finalize before they run out. "
                "When satisfied or nearing budget, explicitly call biosyn_finalize(candidate_id=...) on one evaluated design. "
                "Finalizing may select an infeasible design if the goal remains unmet; report that honestly. "
                "Calling biosyn_submit does NOT end the episode. Finalization does. "
                "Write a short interpretation of computational scope; never claim wet-lab validation."))
        for round_number in (1, 2):
            planner.run_round(round_number, stop_when=lambda: self.summary()["final"] is not None)
            if self.summary()["final"] is not None:
                break
        self.store.write("episode-result.json", self.summary())


if __name__ == "__main__":
    Episode().run()
