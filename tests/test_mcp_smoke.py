"""Smoke test: real MCP client -> real server subprocess over stdio. No LLM."""

import asyncio

from mcp import Client

from equipment.agent.mcp_client import result_payload, server_params


async def _list_and_ping():
    async with Client(server_params()) as client:
        listing = await client.list_tools()
        result = await client.call_tool("ping", {"message": "hi"})
        return [t.name for t in listing.tools], result


def test_server_registers_and_answers_ping():
    names, result = asyncio.run(_list_and_ping())
    assert "ping" in names
    assert not result.is_error
    assert result_payload(result) in ("pong: hi", {"result": "pong: hi"})
