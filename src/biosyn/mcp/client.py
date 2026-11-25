from __future__ import annotations

import json
import threading
import time
import uuid
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from datetime import timedelta

import anyio
import jsonschema
from anyio.from_thread import start_blocking_portal
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from biosyn.config import ServerConfig, expand


class ToolError(RuntimeError):
    pass


class BudgetExceeded(ToolError):
    pass


class MCPHub:
    def __init__(self, servers: dict[str, ServerConfig]):
        self.servers = servers
        self.sessions = {}
        self.catalog = {}

    @asynccontextmanager
    async def lifespan(self):
        async with AsyncExitStack() as stack:
            for name, config in self.servers.items():
                if not config.enabled:
                    continue
                if config.transport == "stdio":
                    command = [expand(item) for item in config.command]
                    params = StdioServerParameters(command=command[0], args=command[1:],
                        env={k: expand(v) for k, v in config.env.items()}, cwd=config.cwd)
                    streams = await stack.enter_async_context(stdio_client(params))
                else:
                    import os
                    headers = {k: os.environ[v] for k, v in config.headers_env.items()}
                    streams = await stack.enter_async_context(streamablehttp_client(
                        expand(config.url), headers=headers, timeout=timedelta(seconds=config.timeout),
                        sse_read_timeout=timedelta(seconds=config.timeout)))
                session = await stack.enter_async_context(ClientSession(
                    streams[0], streams[1], read_timeout_seconds=timedelta(seconds=config.timeout)))
                with anyio.fail_after(config.timeout):
                    await session.initialize()
                    cursor = None
                    while True:
                        page = await session.list_tools(cursor=cursor)
                        for tool in page.tools:
                            key = f"{name}.{tool.name}"
                            if key in self.catalog:
                                raise ToolError(f"Duplicate tool {key}")
                            self.catalog[key] = tool.model_dump(mode="json", exclude_none=True)
                        cursor = page.nextCursor
                        if not cursor:
                            break
                self.sessions[name] = session
            yield self

    async def call(self, name: str, arguments: dict):
        if name not in self.catalog:
            raise ToolError(f"Unknown or disabled MCP tool: {name}")
        jsonschema.validate(arguments, self.catalog[name]["inputSchema"])
        server, tool = name.split(".", 1)
        with anyio.fail_after(self.servers[server].timeout):
            result = await self.sessions[server].call_tool(tool, arguments)
        raw = result.model_dump(mode="json", exclude_none=True)
        if result.isError:
            raise ToolError(json.dumps(raw, ensure_ascii=False))
        if result.structuredContent is not None:
            data = result.structuredContent
        else:
            # Keep all blocks in the receipt; parse JSON only when unambiguous.
            texts = [block.text for block in result.content if block.type == "text"]
            data = {"content": raw.get("content", [])}
            if len(texts) == 1:
                try:
                    data = json.loads(texts[0])
                except json.JSONDecodeError:
                    data = {"text": texts[0], "content": raw.get("content", [])}
        if isinstance(data, dict) and (data.get("status") in {"error", "failed"} or
                                      data.get("success") is False or data.get("error")):
            raise ToolError(json.dumps(raw, ensure_ascii=False))
        return data, raw


class ToolGateway:
    """Synchronous facade with persistent async sessions, usable from Biomni's REPL thread."""

    def __init__(self, servers, store, max_calls=40):
        self.hub = MCPHub(servers)
        self.store = store
        self.max_calls = max_calls
        self.calls = 0
        self._lock = threading.Lock()

    def __enter__(self):
        self._stack = ExitStack()
        try:
            self.portal = self._stack.enter_context(start_blocking_portal())
            self._stack.enter_context(self.portal.wrap_async_context_manager(self.hub.lifespan()))
            self.catalog = self.hub.catalog
            self.store.write("tools.json", self.catalog)
            return self
        except BaseException:
            self._stack.close()
            raise

    def __exit__(self, *args):
        # A caller's exception is not a transport failure. Close sessions normally
        # so AnyIO task groups do not re-wrap that exception several times.
        self._stack.close()
        return False

    def call(self, name: str, arguments: dict):
        with self._lock:
            if self.calls >= self.max_calls:
                self.store.event("tool_budget_exhausted", tool=name, limit=self.max_calls)
                raise BudgetExceeded(f"Tool call budget exhausted ({self.max_calls})")
            self.calls += 1
        call_id = uuid.uuid4().hex
        start = time.monotonic()
        self.store.event("tool_started", call_id=call_id, tool=name, arguments=arguments)
        try:
            data, raw = self.portal.call(self.hub.call, name, arguments)
            receipt = {"call_id": call_id, "tool": name, "arguments": arguments,
                       "elapsed_seconds": time.monotonic() - start, "data": data, "raw": raw}
            self.store.event("tool_succeeded", **receipt)
            return receipt
        except Exception as exc:
            self.store.event("tool_failed", call_id=call_id, tool=name,
                             elapsed_seconds=time.monotonic() - start,
                             error=f"{type(exc).__name__}: {exc}")
            raise
