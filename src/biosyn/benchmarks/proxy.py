"""A per-episode Unix-socket gateway restricted to one host-selected model/endpoint."""
import json
import os
import socketserver
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler

import httpx


@contextmanager
def chat_proxy(path, api_key, log_path, max_calls=24, max_tokens=8192, max_total_tokens=None,
               *, endpoint, model, timeout=240):
    lock = threading.RLock()
    state = {"calls": 0, "total_tokens": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, body):
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                with lock, log_path.open("a") as handle:
                    handle.write(json.dumps({"event": "client_disconnected", "http_status": status}) + "\n")

        def do_POST(self):
            started = time.monotonic()
            if self.path != "/v1/chat/completions":
                self.reply(403, b'{"error":"Only the configured chat endpoint is allowed"}')
                return
            size = int(self.headers.get("Content-Length", 0))
            if not 0 < size <= 4_000_000:
                self.reply(413, b'{"error":"Request size out of bounds"}')
                return
            body = self.rfile.read(size)
            try:
                data = json.loads(body)
                if (data.get("model") != model or data.get("stream") or
                        not 1 <= data.get("max_completion_tokens", data.get("max_tokens", 0)) <= max_tokens):
                    raise ValueError("Unexpected model, streaming request or token limit")
            except (ValueError, TypeError) as exc:
                self.reply(400, json.dumps({"error": str(exc)}).encode())
                return
            with lock:
                if state["calls"] >= max_calls or (max_total_tokens is not None and state["total_tokens"] >= max_total_tokens):
                    self.reply(429, b'{"error":"Per-episode LLM call budget exhausted"}')
                    return
                state["calls"] += 1
                call = state["calls"]
            try:
                response = httpx.post(endpoint,
                    content=body, headers={"Content-Type": "application/json",
                                          "Authorization": "Bearer " + api_key}, timeout=timeout)
                result = response.json()
                with lock, log_path.open("a") as handle:
                    state["total_tokens"] += (result.get("usage") or {}).get("total_tokens", 0)
                    handle.write(json.dumps({"call": call, "http_status": response.status_code,
                        "model": result.get("model"), "usage": result.get("usage"),
                        "elapsed_seconds": time.monotonic() - started,
                        "requested_output_tokens": data.get("max_completion_tokens", data.get("max_tokens")),
                        "finish_reasons": [x.get("finish_reason") for x in result.get("choices", [])]}) + "\n")
                self.reply(response.status_code, response.content)
            except httpx.HTTPError as exc:
                with lock, log_path.open("a") as handle:
                    handle.write(json.dumps({"call": call, "error": type(exc).__name__,
                        "elapsed_seconds": time.monotonic() - started}) + "\n")
                self.reply(502, json.dumps({"error": type(exc).__name__}).encode())

    class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
        daemon_threads = True

    server = Server(str(path), Handler)
    os.chmod(path, 0o600)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=10)
        path.unlink(missing_ok=True)


def deepseek_proxy(path, api_key, log_path, max_calls=24, max_tokens=8192, max_total_tokens=None):
    return chat_proxy(path, api_key, log_path, max_calls, max_tokens, max_total_tokens,
                      endpoint="https://api.deepseek.com/chat/completions", model="deepseek-v4-flash")
