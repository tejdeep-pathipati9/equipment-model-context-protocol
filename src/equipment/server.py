"""MCP server (stdio transport) + composition root.

Run with `python -m equipment.server` (or `uv run server`). Wrappers here only pass arguments
through; business logic lives in `equipment.tools`. This is the only place that reads config
(EQUIP_DB_PATH, AS_OF_DATE) and wires the database and "today" into the tool functions.
"""

import asyncio
import os
import sqlite3
import sys
from collections.abc import Generator
from contextlib import closing, contextmanager
from datetime import date
from typing import Any

from mcp.server.mcpserver import MCPServer

from equipment import tools
from equipment.data.seed import db_path
from equipment.data.sqlite_repo import SqliteRepository

mcp = MCPServer("equipment")


def as_of() -> date:
    """AS_OF_DATE if set (demo/CI), else the real date."""
    value = os.environ.get("AS_OF_DATE")
    return date.fromisoformat(value) if value else date.today()


@contextmanager
def repository() -> Generator[SqliteRepository]:
    """A fresh connection per call: tools may run in a worker thread, and a SQLite
    connection can't be shared across threads by default."""
    path = db_path()
    if not path.exists():
        raise FileNotFoundError(f"database {path} not found; run `uv run seed` first")
    with closing(sqlite3.connect(path)) as conn:
        yield SqliteRepository(conn)


@mcp.tool()
def ping(message: str = "hello") -> str:
    """Connectivity check: echoes the message back."""
    return f"pong: {message}"


@mcp.tool()
def get_employee_info(employee_id: str) -> dict[str, Any]:
    """Look up an employee's role, tenure in months, and equipment on file (item, issue date,
    age in months). Returns {"error": "EMPLOYEE_NOT_FOUND", ...} for an unknown ID."""
    with repository() as repo:
        return tools.get_employee_info(repo, employee_id, as_of())


def main() -> None:
    # stdout carries the MCP protocol, so human-readable status goes to stderr only.
    registered = asyncio.run(mcp.list_tools())
    log(f"equipment MCP server ready on stdio, {len(registered)} tool(s):")
    for tool in registered:
        log(f"  - {tool.name}")
    log(f"db={db_path()} as_of={as_of()}")
    log("waiting for a client (JSON-RPC on stdin); Ctrl+C to stop")
    try:
        mcp.run()  # stdio
    except KeyboardInterrupt:
        log("stopped")


def log(message: str) -> None:
    print(f"[server] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
