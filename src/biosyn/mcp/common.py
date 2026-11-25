import argparse
import re
from hashlib import sha256


def sequence(value: str, alphabet: str, max_length: int = 100000) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= max_length or not re.fullmatch(f"[{alphabet}]+", value):
        raise ValueError(f"Expected 1..{max_length} uppercase characters from {alphabet}")
    return value


def digest(value: str) -> str:
    return sha256(value.encode()).hexdigest()


def serve(server):
    parser = argparse.ArgumentParser()
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    server.settings.host = args.host
    server.settings.port = args.port
    server.run(transport=args.transport)
