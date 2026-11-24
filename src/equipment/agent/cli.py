"""Agent entrypoint + composition root (`uv run agent`)."""

import argparse
import asyncio
import json
import sys
from pathlib import Path

import anthropic

from equipment.agent.llm import FoundryClient
from equipment.agent.mcp_client import ServerUnavailable, connect
from equipment.agent.react import Request, run_agent
from equipment.agent.trace import format_decision, format_step

SCENARIOS = Path("scenarios/demo.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent",
        description="Approve, deny or escalate an IT equipment request.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--scenario", type=int, metavar="N", help="run demo scenario N from scenarios/demo.json"
    )
    source.add_argument("--employee", metavar="ID", help="employee ID from request metadata")
    parser.add_argument("--text", help="free-text request (required with --employee)")
    return parser


def load_scenario(number: int) -> Request:
    scenarios = json.loads(SCENARIOS.read_text())
    if not 1 <= number <= len(scenarios):
        raise SystemExit(f"scenario must be 1-{len(scenarios)}")
    scenario = scenarios[number - 1]
    return Request(scenario["employee_id"], scenario["text"])


async def run(request: Request) -> int:
    llm = FoundryClient.from_env()
    print(f"Request:     {request.employee_id}: {request.text}\n")
    steps = 0

    def show(step):
        nonlocal steps
        steps += 1
        print(format_step(steps, step), end="\n\n", flush=True)

    async with connect() as mcp:
        try:
            result = await run_agent(llm, mcp, request, on_step=show)
        except anthropic.APIStatusError as error:
            print(f"Foundry error {error.status_code}: {error.message}", file=sys.stderr)
            return 1
        except anthropic.APIConnectionError:
            print("cannot reach Foundry: check the endpoint setting in .env", file=sys.stderr)
            return 1
    print(format_decision(result.decision))
    return 0


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.employee and not args.text:
        parser.error("--text is required with --employee")
    request = load_scenario(args.scenario) if args.scenario else Request(args.employee, args.text)
    try:
        sys.exit(asyncio.run(run(request)))
    except ServerUnavailable as error:
        raise SystemExit(str(error)) from None


if __name__ == "__main__":
    main()
