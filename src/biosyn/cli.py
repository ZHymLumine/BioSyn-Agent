import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from biosyn.agent.runtime import DesignRuntime
from biosyn.agent.state import RunStore, TaskSpec
from biosyn.config import load_config
from biosyn.mcp.client import ToolGateway


def main():
    parser = argparse.ArgumentParser(description="BioSyn-Agent on Biomni")
    parser.add_argument("--env-file", type=Path)
    commands = parser.add_subparsers(dest="action", required=True)
    export = commands.add_parser("export-biodesignbench")
    export.add_argument("--run-dir", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    for name in ("run", "evaluate", "catalog", "tool"):
        command = commands.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--run-dir", type=Path, required=True)
        if name in ("run", "evaluate"):
            command.add_argument("--task", type=Path, required=True)
        if name == "tool":
            command.add_argument("name")
            command.add_argument("--args", required=True, help="JSON object")
    args = parser.parse_args()
    if args.env_file:
        if not args.env_file.is_file():
            parser.error("Credential env file does not exist")
        load_dotenv(args.env_file, override=False)
    if args.action == "export-biodesignbench":
        from biosyn.integrations.biodesignbench import export_output
        export_output(args.run_dir, args.output)
        print(args.output.resolve())
        return 0
    config = load_config(args.config)
    if args.action in ("run", "evaluate"):
        task = TaskSpec.model_validate_json(args.task.read_text())
        state = DesignRuntime(task, config, args.run_dir,
                              mode="biomni" if args.action == "run" else "evaluate").run()
        print(json.dumps({"status": state.status, "best_id": state.best_id,
                          "report": str(args.run_dir.resolve() / "report.json")}, indent=2))
        return 0 if state.status == "passed" else 2
    store = RunStore(args.run_dir)
    with ToolGateway(config.servers, store, max_calls=1) as gateway:
        output = gateway.catalog if args.action == "catalog" else gateway.call(args.name, json.loads(args.args))
        print(json.dumps(output, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
