"""MCP server (Streamable HTTP or stdio) + composition root.

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
from datetime import UTC, date, datetime
from typing import Any

from mcp.server.mcpserver import MCPServer

from equipment import config, tools
from equipment.config import Transport
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
def get_employee_info(employee_id: str) -> dict[str, Any]:
    """Look up an employee's role, tenure in months, and equipment on file (item, issue date,
    age in months). Returns {"error": "EMPLOYEE_NOT_FOUND", ...} for an unknown ID."""
    with repository() as repo:
        return tools.get_employee_info(repo, employee_id, as_of())


@mcp.tool()
def get_policy_limits(role: str) -> dict[str, Any]:
    """Look up what a role may hold: max quantity (0 = not eligible) and refresh interval in
    months per item. Returns {"error": "UNKNOWN_ROLE", ...} for an unknown role."""
    with repository() as repo:
        return tools.get_policy_limits(repo, role)


@mcp.tool()
def check_request_eligibility(employee_id: str, item: str) -> dict[str, Any]:
    """Check whether policy and records allow an employee one unit of an item. Returns status
    (eligible, not_eligible, needs_review), reason_code, basis (quantity = one more unit,
    refresh = replaces the oldest unit) and the numbers used. If nothing in the catalog matches,
    pass the item as the employee wrote it."""
    with repository() as repo:
        return tools.check_request_eligibility(repo, employee_id, item, as_of())


@mcp.tool()
def flag_for_human_review(
    employee_id: str, request: str, reason: str, reason_code: str
) -> dict[str, Any]:
    """Escalate a request to a human. Saves a record, so call it at most once per request.
    reason_code: UNKNOWN_EMPLOYEE, VAGUE_REQUEST, ITEM_NOT_IN_CATALOG, RECORDS_INCOMPLETE,
    CLAIM_CONTRADICTS_RECORDS, EXCEPTION_REQUEST, REFLECTION_FAILED or AGENT_STEP_LIMIT.
    Returns the escalation_id and status."""
    with repository() as repo:
        return tools.flag_for_human_review(
            repo, employee_id, request, reason, reason_code, datetime.now(UTC)
        )


def main() -> None:
    # With stdio, stdout carries the MCP protocol, so status goes to stderr only.
    transport = config.transport()
    registered = asyncio.run(mcp.list_tools())
    log(f"equipment MCP server on {transport}, {len(registered)} tool(s):")
    for tool in registered:
        log(f"  - {tool.name}")
    log(f"db={db_path()} as_of={as_of()}")
    try:
        if transport is Transport.STDIO:
            log("waiting for a client (JSON-RPC on stdin); Ctrl+C to stop")
            mcp.run("stdio")
        else:
            log(f"listening on http://{config.host()}:{config.port()}/mcp; Ctrl+C to stop")
            mcp.run("streamable-http", host=config.host(), port=config.port())
    except KeyboardInterrupt:
        log("stopped")


def log(message: str) -> None:
    print(f"[server] {message}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
