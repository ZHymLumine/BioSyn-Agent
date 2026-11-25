import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from biosyn.agent.state import RunStore
from biosyn.config import ServerConfig
from biosyn.mcp.client import ToolError, ToolGateway

FIXTURE = str(Path(__file__).with_name("fixture_mcp.py"))


def test_persistent_session_all_content_blocks_and_timeout(tmp_path):
    config = ServerConfig(command=[sys.executable, FIXTURE], timeout=10)
    with ToolGateway({"fixture": config}, RunStore(tmp_path / "protocol"), max_calls=6) as gateway:
        a = gateway.call("fixture.count", {})["data"]
        b = gateway.call("fixture.count", {})["data"]
        assert a["pid"] == b["pid"] and b["count"] == a["count"] + 1
        content = gateway.call("fixture.blocks", {})["raw"]["content"]
        assert [b["text"] for b in content] == ["first", "second"]
        with pytest.raises(ToolError, match="deliberate"):
            gateway.call("fixture.application_error", {})
        config.timeout = 0.1
        with pytest.raises(TimeoutError):
            gateway.call("fixture.slow", {})


def test_streamable_http(tmp_path):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    process = subprocess.Popen([sys.executable, FIXTURE, "--transport", "streamable-http", "--port", str(port)],
                               stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    url = f"http://127.0.0.1:{port}/mcp"
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                httpx.get(url, timeout=0.2)
                break
            except httpx.HTTPError:
                if process.poll() is not None or time.monotonic() >= deadline:
                    raise RuntimeError("HTTP test server did not start")
                time.sleep(0.05)
        config = ServerConfig(transport="streamable-http", url=url)
        with ToolGateway({"fixture": config}, RunStore(tmp_path / "http"), max_calls=1) as gateway:
            assert gateway.call("fixture.count", {})["data"]["count"] == 1
    finally:
        process.terminate()
        process.communicate(timeout=10)
