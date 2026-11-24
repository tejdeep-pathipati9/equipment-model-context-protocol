"""Smoke test: real MCP client -> real server subprocess over stdio. No LLM."""

import asyncio
import sqlite3
from contextlib import closing

import pytest
from mcp import Client

from equipment.agent.mcp_client import result_payload, server_params
from equipment.data.seed import create_schema, seed_data


@pytest.fixture
def server_db(db_path, monkeypatch):
    """Seeded DB file the server subprocess can see (EQUIP_DB_PATH, AS_OF_DATE forwarded)."""
    with closing(sqlite3.connect(db_path)) as conn, conn:  # `with conn` commits
        create_schema(conn)
        seed_data(conn)
    monkeypatch.setenv("AS_OF_DATE", "2026-12-01")
    return db_path


async def _session(calls):
    async with Client(server_params()) as client:
        listing = await client.list_tools()
        results = [await client.call_tool(name, args) for name, args in calls]
        return [t.name for t in listing.tools], results


def test_server_registers_tools(server_db):
    names, _ = asyncio.run(_session([]))
    assert names == ["get_employee_info"]


def test_get_employee_info_over_mcp(server_db):
    _, [result] = asyncio.run(_session([("get_employee_info", {"employee_id": "E001"})]))

    assert not result.is_error
    info = result_payload(result)
    assert info["role"] == "employee"
    assert info["tenure_months"] == 44  # proves AS_OF_DATE reached the server
    assert [unit["item"] for unit in info["equipment"]] == ["laptop", "monitor"]


def test_get_employee_info_unknown_id_over_mcp(server_db):
    _, [result] = asyncio.run(_session([("get_employee_info", {"employee_id": "E999"})]))
    assert result_payload(result)["error"] == "EMPLOYEE_NOT_FOUND"
