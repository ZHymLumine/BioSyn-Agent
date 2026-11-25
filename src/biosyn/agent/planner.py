"""Use Biomni A1's retriever, LangGraph ReAct graph, REPL, and know-how loader."""
import json
import os
import re
import sys
import types
from pathlib import Path

import jsonschema
from langchain_core.messages import HumanMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_openai import ChatOpenAI

from biosyn.agent.state import Candidate


MAX_CALLS_WITHOUT_ARTIFACT_PER_ROUND = 8


class UsageRecorder(BaseCallbackHandler):
    def __init__(self, store):
        self.store = store

    def on_llm_end(self, response, **kwargs):
        for generations in response.generations:
            for generation in generations:
                message = getattr(generation, "message", None)
                self.store.event("llm_completed", usage=getattr(message, "usage_metadata", None) or {},
                                 response_metadata=getattr(message, "response_metadata", {}))

    def on_llm_error(self, error, **kwargs):
        self.store.event("llm_failed", error=f"{type(error).__name__}: {error}")


def register(agent, name, description, schema, function):
    """Register a known JSON schema without asking an LLM to invent its signature."""
    module_name = "biosyn_tools"
    module = sys.modules.setdefault(module_name, types.ModuleType(module_name))
    if name in getattr(agent, "_custom_functions", {}):
        raise ValueError(f"Duplicate Biomni tool {name}")

    def wrapped(**kwargs):
        jsonschema.validate(kwargs, schema)
        return function(**kwargs)

    wrapped.__name__ = name
    wrapped.__module__ = module_name
    wrapped.__doc__ = description + "\nInput JSON schema: " + json.dumps(schema)
    setattr(module, name, wrapped)
    required = schema.get("required", [])
    params = [{"name": key, "type": value.get("type", "object"),
               "description": value.get("description", ""), "default": value.get("default")}
              for key, value in schema.get("properties", {}).items()]
    api = {"name": name, "description": wrapped.__doc__, "module": module_name,
           "required_parameters": [p for p in params if p["name"] in required],
           "optional_parameters": [p for p in params if p["name"] not in required]}
    agent.module2api.setdefault(module_name, []).append(api)
    agent._custom_functions = getattr(agent, "_custom_functions", {})
    agent._custom_functions[name] = wrapped
    agent._custom_tools = getattr(agent, "_custom_tools", {})
    agent._custom_tools[name] = {"name": name, "description": wrapped.__doc__, "module": module_name}


class BiomniPlanner:
    def __init__(self, config, runtime, llm=None, submission_schema=None, instructions=None,
                 submission_description=None):
        from biomni.agent import A1
        from biomni.tool.tool_registry import ToolRegistry
        if config.llm is None:
            raise ValueError("Biomni mode requires an explicit llm configuration")
        settings = config.llm
        key = os.environ.get(settings.api_key_env)
        if llm is None and not key:
            raise ValueError(f"Required credential variable {settings.api_key_env} is unset")
        self.runtime = runtime
        self.config = config
        self.instructions = instructions
        self.agent = A1(path=str(runtime.store.path / "biomni_data"), llm=settings.model,
            source="Custom", base_url=settings.base_url, api_key=key or "test-injected",
            expected_data_lake_files=[], use_tool_retriever=config.use_tool_retriever,
            timeout_seconds=config.code_timeout)
        transport = {}
        if settings.api_socket:
            import httpx
            transport = {"http_client": httpx.Client(transport=httpx.HTTPTransport(uds=settings.api_socket)),
                         "http_async_client": httpx.AsyncClient(transport=httpx.AsyncHTTPTransport(uds=settings.api_socket))}
        self.agent.llm = llm if llm is not None else ChatOpenAI(
            model=settings.model, base_url=settings.base_url, api_key=key,
            temperature=settings.temperature, max_tokens=settings.max_tokens,
            timeout=settings.timeout, max_retries=0, extra_body=settings.extra_body,
            callbacks=[UsageRecorder(runtime.store)],
            stop=["</execute>", "</solution>"], **transport)
        if config.biomni_tool_modules is not None:
            unknown = set(config.biomni_tool_modules) - self.agent.module2api.keys()
            if unknown:
                raise ValueError(f"Unknown Biomni tool modules: {sorted(unknown)}")
            self.agent.module2api = {k: v for k, v in self.agent.module2api.items()
                                     if k in config.biomni_tool_modules}
            excluded = set(config.biomni_excluded_tools)
            known_tools = {tool["name"] for tools in self.agent.module2api.values() for tool in tools}
            unknown_tools = excluded - known_tools
            if unknown_tools:
                raise ValueError(f"Unknown excluded Biomni tools: {sorted(unknown_tools)}")
            if excluded:
                self.agent.module2api = {
                    module: [tool for tool in tools if tool["name"] not in excluded]
                    for module, tools in self.agent.module2api.items()
                }
            # A lightweight deployment must not advertise the full Biomni E1 environment.
            self.agent.library_content_dict = {}
            self.agent.data_lake_dict = {}
            if not config.preserve_native_biomni_knowhow:
                self.agent.know_how_loader.documents = {}
        register(self.agent, "biosyn_submit", submission_description or "Submit a new candidate for tool-based evaluation; returns measured constraints and scores.",
            {"type": "object", "properties": {"candidate": submission_schema or Candidate.model_json_schema()},
             "required": ["candidate"], "additionalProperties": False}, runtime.submit)
        register(self.agent, "biosyn_get_state", "Read candidate history, tool budget, failures, and best accepted design.",
            {"type": "object", "properties": {}, "additionalProperties": False}, runtime.summary)
        if callable(getattr(runtime, "finalize", None)):
            register(self.agent, "biosyn_finalize", "Explicitly select an evaluated immutable design and end the design episode.",
                {"type": "object", "properties": {"candidate_id": {"type": "string"}},
                 "required": ["candidate_id"], "additionalProperties": False}, runtime.finalize)
        if config.register_runtime_tools:
            for qualified, tool in runtime.gateway.catalog.items():
                name = re.sub(r"[^A-Za-z0-9_]", "_", qualified.replace(".", "__", 1))
                def call(_qualified=qualified, **arguments):
                    return runtime.gateway.call(_qualified, arguments)
                register(self.agent, name, tool.get("description", qualified), tool["inputSchema"], call)
        runtime.store.event(
            "planner_capabilities",
            biomni_tool_modules=sorted(self.agent.module2api),
            excluded_biomni_tools=sorted(config.biomni_excluded_tools),
            native_knowhow=config.preserve_native_biomni_knowhow,
            runtime_tools_registered=config.register_runtime_tools,
            runtime_tool_count=len(runtime.gateway.catalog) if config.register_runtime_tools else 0,
        )
        self.agent.tool_registry = ToolRegistry(self.agent.module2api)
        for path in config.skills:
            document = Path(path)
            content = document.read_text()
            self.agent.know_how_loader.add_custom_document(
                doc_id=str(document.resolve()), name=document.stem,
                description=f"User-configured workflow guidance: {document.name}",
                content=content)
            # The checked-out A1.configure expects this field; upstream's
            # add_custom_document currently omits it (loader-created docs have it).
            self.agent.know_how_loader.documents[str(document.resolve())]["content_without_metadata"] = (
                self.agent.know_how_loader._strip_metadata(content))
        self.agent.configure()

    def run_round(self, round_number: int, *, stop_when=None, stop_for_budget=None):
        runtime = self.runtime
        initial_state = runtime.summary()
        watchdog_fields = {"history", "final", "remaining_llm_requests"}
        watchdog_enabled = watchdog_fields.issubset(initial_state)
        default_instructions = (
            "You are BioSyn-Agent, operating through Biomni. Solve the user request below. "
            "Use the registered tools to inspect, design, predict and evaluate. "
            "Submit every proposed design via biosyn_submit(candidate=...). "
            "Only that tool's evaluation determines acceptance. Do not fabricate measurements. "
            "A rejected candidate should be revised under a new id with parent_id set. "
            "Inspect returned errors and constraint violations, then try again within the budget. "
            "Use biosyn_get_state to inspect remaining budget. Stop when a design passes or the budget is exhausted. "
            "Phenotype entries are requested properties, never measured evidence. "
            "Return a concise final explanation with candidate ids and remaining uncertainties.\n"
        )
        recovery = ""
        if watchdog_enabled and round_number > 1 and not initial_state["history"]:
            recovery = (
                "\nSUBMISSION RECOVERY: The previous planning round ended without a submitted artifact. "
                "Do not inspect more catalog entries or copy fixed DNA parts before submission. Use the "
                "already selected BioSyn scientific helper tools, construct only the public decision_schema "
                "fields, and call biosyn_submit in this round. An imperfect or invalid checkpoint candidate "
                "is acceptable because its real evaluator feedback enables the next revision.\n"
            )
        prompt = ((self.instructions or default_instructions) + "\n" +
            recovery +
            f"TASK:\n{runtime.state.task.model_dump_json(indent=2)}\n"
            f"CURRENT STATE / REFLECTION INPUT:\n{json.dumps(runtime.summary(), ensure_ascii=False)}"
        )
        agent = self.agent
        agent.critic_count = 0
        agent.user_task = prompt
        agent.log = []
        if agent.use_tool_retriever:
            resources = agent._prepare_resources_for_retrieval(prompt)
            agent.update_system_prompt_with_selected_resources(resources)
            runtime.store.event("retrieval", round=round_number, selected=resources)
        graph_config = {"recursion_limit": self.config.recursion_limit,
                        "configurable": {"thread_id": f"{runtime.state.run_id}:{round_number}"}}
        final = None
        try:
            for final in agent.app.stream({"messages": [HumanMessage(content=prompt)], "next_step": None},
                                          stream_mode="values", config=graph_config):
                last = final["messages"][-1]
                runtime.store.event("biomni_message", round=round_number, role=last.type,
                                    content=last.content,
                                    usage=getattr(last, "usage_metadata", None),
                                    response_metadata=getattr(last, "response_metadata", {}))
                if stop_when is not None and stop_when():
                    runtime.store.event("planner_stopped", reason="submission_finalized", round=round_number)
                    break
                # Finish an already generated tool action before reserving the
                # remaining model budget for selection. Never discard a pending submit.
                safe_to_stop = (str(last.content).lstrip().startswith("<observation>")
                                or final.get("next_step") != "execute")
                if (stop_for_budget is not None and safe_to_stop
                        and stop_for_budget()):
                    runtime.store.event("planner_stopped", reason="finalization_reserve", round=round_number)
                    break
                if watchdog_enabled:
                    current_state = runtime.summary()
                    if watchdog_fields.issubset(current_state):
                        calls_used = max(
                            0,
                            initial_state["remaining_llm_requests"]
                            - current_state["remaining_llm_requests"],
                        )
                        if (
                            safe_to_stop
                            and current_state["final"] is None
                            and not current_state["history"]
                            and calls_used >= MAX_CALLS_WITHOUT_ARTIFACT_PER_ROUND
                        ):
                            runtime.store.event(
                                "planner_stopped",
                                reason="no_artifact_call_limit",
                                round=round_number,
                                calls_used=calls_used,
                            )
                            break
        finally:
            if final:
                agent._conversation_state = final
                runtime.store.write(f"conversation-{round_number}.json",
                    [{"role": m.type, "content": m.content} for m in final["messages"]])
        return final["messages"][-1].content if final else ""
