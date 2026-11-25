"""Native deployment of the upstream CPU interface tool; scientific code is reused unchanged."""
import argparse
import asyncio
import os
import sys
from dataclasses import replace
from pathlib import Path

import yaml
from mcp.server import Server
from mcp.server.stdio import stdio_server


async def run(root: Path, scratch: Path):
    from protein_design_mcp.app import ServerApp
    from protein_design_mcp.dispatch.env import EnvDispatcher
    from protein_design_mcp.manifest.registry import ToolRegistry
    from protein_design_mcp.manifest.schema import parse_manifest
    root = root.resolve(strict=True)
    script = (root / "scripts/engines/interface_residues.py").resolve(strict=True)
    manifest_path = root / "src/protein_design_mcp/manifests/run_interface_residues.yaml"
    manifest = parse_manifest(yaml.safe_load(manifest_path.read_text()))
    # The official deployment uses /app and micromamba inside a container.
    # This explicit native profile changes only the interpreter/script location.
    manifest = replace(manifest, engine=replace(manifest.engine, entry=(sys.executable, str(script))))
    scratch.mkdir(parents=True, exist_ok=True)
    os.environ["PROTEIN_MCP_RESULTS_DIR"] = str(scratch.resolve() / "results")
    os.environ["PROTEIN_MCP_KEEP_WORKDIR"] = "1"
    app = ServerApp(ToolRegistry([manifest], device="cpu"),
                    dispatcher=EnvDispatcher(runner=None, scratch_root=scratch.resolve()))
    server = Server("BioSyn protein-design-mcp native CPU")

    @server.list_tools()
    async def list_tools():
        return await app.list_tools()

    @server.call_tool()
    async def call_tool(name, arguments):
        return await app.call_tool(name, arguments)

    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args.root, args.scratch))


if __name__ == "__main__":
    main()
