"""Protocol test fixture only. Never used by deployment configurations."""
import asyncio
import os

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from biosyn.mcp.common import serve

server = FastMCP("Protocol test fixture", log_level="WARNING")
counter = 0


@server.tool()
def count() -> dict:
    global counter
    counter += 1
    return {"pid": os.getpid(), "count": counter}


@server.tool()
def blocks() -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text="first"),
                                   TextContent(type="text", text="second")])


@server.tool()
def application_error() -> dict:
    return {"error": "deliberate test failure"}


@server.tool()
async def slow() -> dict:
    await asyncio.sleep(5)
    return {"done": True}


if __name__ == "__main__":
    serve(server)
