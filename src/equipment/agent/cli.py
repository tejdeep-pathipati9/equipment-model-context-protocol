"""Agent entrypoint + composition root (`uv run agent`)."""

import argparse
import sys


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


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.employee and not args.text:
        parser.error("--text is required with --employee")
    sys.exit("agent: not implemented yet (PLAN.md build step 9)")


if __name__ == "__main__":
    main()
