"""MCP client: connect over Streamable HTTP (default) or stdio, list tools, call a tool.

Used by the agent, and runnable on its own as a connectivity check:

    uv run python -m equipment.agent.mcp_client                                          # list
    uv run python -m equipment.agent.mcp_client get_employee_info '{"employee_id": "E001"}'  # call
"""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any

from anthropic.types import ToolParam
from mcp import Client, StdioServerParameters, Tool

from equipment import config
from equipment.config import Transport

# The stdio client only forwards a small allowlist of env vars, so pass ours explicitly.
FORWARDED_ENV = ("EQUIP_DB_PATH", "AS_OF_DATE")


def server_params() -> StdioServerParameters:
    """How to launch the server: same interpreter, as a module, with our config vars."""
    env = {key: os.environ[key] for key in FORWARDED_ENV if key in os.environ}
    env["MCP_TRANSPORT"] = Transport.STDIO  # the server's own default is HTTP
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


class ServerUnavailable(Exception):
    """The HTTP server can't be reached."""


def connection() -> StdioServerParameters | str:
    """How to reach the server: launch settings for stdio, or a URL."""
    return server_params() if config.transport() is Transport.STDIO else config.url()


@asynccontextmanager
async def connect() -> AsyncGenerator[Client]:
    """A connected client, whichever transport is configured."""
    target = connection()
    async with AsyncExitStack() as stack:
        try:
            client = await stack.enter_async_context(Client(target))
        except Exception as error:
            if not isinstance(target, str):
                raise
            raise ServerUnavailable(
                f"no MCP server at {target}; start it with `uv run server`"
            ) from error
        yield client


def anthropic_tools(tools: list[Tool]) -> list[ToolParam]:
    """MCP tools in the format the Anthropic API expects."""
    return [
        {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
        for t in tools
    ]


async def run(tool: str | None, arguments: dict[str, Any]) -> None:
    target = connection()
    if isinstance(target, str):
        print(f"[client] connecting to {target}")
    else:
        print(f"[client] launching server: {target.command} {' '.join(target.args)}")
    async with connect() as client:
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
    parser = argparse.ArgumentParser(description="Connect to the equipment MCP server.")
    parser.add_argument("tool", nargs="?", help="tool to call (omit to only list tools)")
    parser.add_argument("arguments", nargs="?", default="{}", help="tool arguments as JSON")
    args = parser.parse_args(argv)
    try:
        asyncio.run(run(args.tool, json.loads(args.arguments)))
    except ServerUnavailable as error:
        raise SystemExit(f"[client] {error}") from None


if __name__ == "__main__":
    main()
