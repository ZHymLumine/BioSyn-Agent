"""Run the sequence design example with a configured LLM endpoint."""

import argparse
from pathlib import Path

from dotenv import load_dotenv

from biosyn.agent.runtime import DesignRuntime
from biosyn.agent.state import TaskSpec
from biosyn.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/biomni/local.yaml"))
    parser.add_argument("--task", type=Path, default=Path("examples/tasks/cds_design.json"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/sequence-design"))
    args = parser.parse_args()
    load_dotenv()
    task = TaskSpec.model_validate_json(args.task.read_text())
    state = DesignRuntime(task, load_config(args.config), args.run_dir).run()
    print(f"Status: {state.status}; selected candidate: {state.best_id}")


if __name__ == "__main__":
    main()
