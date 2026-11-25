"""Persistent MCP evaluation with one fixed, hash-checked application profile."""
import argparse

from mcp.server.fastmcp import FastMCP

from biosyn.applications.science import Evaluator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--resources", required=True)
    args = parser.parse_args()
    evaluator = Evaluator(args.profile, args.resources)
    server = FastMCP("BioSyn real-resource application evaluator")

    @server.tool()
    def evaluate_design(design: dict) -> dict:
        """Evaluate one design with the immutable public profile. Returns predictions or simulations, not measurements."""
        return evaluator.evaluate(design)

    @server.tool()
    def describe_resources() -> dict:
        """Read original calibrated gate parameters, model gene ids or fixed sequence context."""
        return evaluator.describe()

    server.run()


if __name__ == "__main__":
    main()
