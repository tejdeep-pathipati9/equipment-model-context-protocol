"""MCP client: spawn the server over stdio, list tools, call a tool.

Used by the agent, and runnable on its own as a connectivity check:

    uv run python -m equipment.agent.mcp_client                      # list tools
    uv run python -m equipment.agent.mcp_client ping '{"message": "hi"}'   # call one
"""

import argparse
import asyncio
import json
import os
import sys
from typing import Any

from mcp import Client, StdioServerParameters

# The stdio client only forwards a small allowlist of env vars, so pass ours explicitly.
FORWARDED_ENV = ("EQUIP_DB_PATH", "AS_OF_DATE")


def server_params() -> StdioServerParameters:
    """How to launch the server: same interpreter, as a module, with our config vars."""
    env = {key: os.environ[key] for key in FORWARDED_ENV if key in os.environ}
    return StdioServerParameters(command=sys.executable, args=["-m", "equipment.server"], env=env)


def result_payload(result: Any) -> Any:
    """Return the tool's output: structured content if present, else the text, parsed if JSON."""
    structured = getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    text = "".join(block.text for block in result.content if getattr(block, "text", None))
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


async def run(tool: str | None, arguments: dict[str, Any]) -> None:
    params = server_params()
    print(f"[client] launching server: {params.command} {' '.join(params.args)}")
    async with Client(params) as client:
        print("[client] connected, session initialized")

        listing = await client.list_tools()
        print(f"[client] server registered {len(listing.tools)} tool(s):")
        for t in listing.tools:
            print(f"  - {t.name}: {t.description}")
            print(f"    input schema: {json.dumps(t.input_schema)}")

        if tool:
            print(f"[client] call_tool({tool!r}, {json.dumps(arguments)})")
            result = await client.call_tool(tool, arguments)
            status = "ERROR" if result.is_error else "ok"
            print(f"[client] result ({status}): {json.dumps(result_payload(result), indent=2)}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Connect to the equipment MCP server over stdio.")
    parser.add_argument("tool", nargs="?", help="tool to call (omit to only list tools)")
    parser.add_argument("arguments", nargs="?", default="{}", help="tool arguments as JSON")
    args = parser.parse_args(argv)
    asyncio.run(run(args.tool, json.loads(args.arguments)))


if __name__ == "__main__":
    main()
