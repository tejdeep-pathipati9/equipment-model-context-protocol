"""MCP server (stdio transport) + composition root.

Run with `python -m equipment.server` (or `uv run server`). Wrappers here only convert arguments
and results; business logic lives in `equipment.tools` (PLAN.md §6).

Build step 3: one trivial tool to prove the server/client plumbing before any business logic.
"""

import asyncio
import sys

from mcp.server.mcpserver import MCPServer

mcp = MCPServer("equipment")


@mcp.tool()
def ping(message: str = "hello") -> str:
    """Connectivity check: echoes the message back."""
    return f"pong: {message}"


def main() -> None:
    # stdout carries the MCP protocol, so human-readable status goes to stderr only.
    tools = asyncio.run(mcp.list_tools())
    log(f"equipment MCP server ready on stdio, {len(tools)} tool(s): {[t.name for t in tools]}")
    log("waiting for a client (JSON-RPC on stdin); Ctrl+C to stop")
    try:
        mcp.run()  # stdio
    except KeyboardInterrupt:
        log("stopped")


def log(message: str) -> None:
    print(f"[server] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
