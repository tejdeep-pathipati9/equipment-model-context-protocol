"""Choosing how the client and server connect (MCP_TRANSPORT and friends)."""

import asyncio

import pytest
from mcp import StdioServerParameters

from equipment import config
from equipment.agent.mcp_client import ServerUnavailable, connect, connection


@pytest.fixture(autouse=True)
def no_mcp_settings(monkeypatch):
    for name in ("MCP_TRANSPORT", "MCP_HOST", "MCP_PORT", "MCP_URL"):
        monkeypatch.delenv(name, raising=False)


def test_default_is_streamable_http_on_localhost():
    assert config.transport() == "streamable-http"
    assert connection() == "http://127.0.0.1:8000/mcp"


def test_stdio_launches_the_server(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")

    target = connection()

    assert isinstance(target, StdioServerParameters)
    assert target.env["MCP_TRANSPORT"] == "stdio"


def test_client_follows_the_server_host_and_port(monkeypatch):
    monkeypatch.setenv("MCP_HOST", "localhost")
    monkeypatch.setenv("MCP_PORT", "9001")

    assert connection() == "http://localhost:9001/mcp"


def test_url_overrides_host_and_port(monkeypatch):
    monkeypatch.setenv("MCP_URL", "https://equipment.example/mcp")
    monkeypatch.setenv("MCP_PORT", "9001")

    assert connection() == "https://equipment.example/mcp"


def test_unknown_transport_names_the_choices(monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "carrier-pigeon")

    with pytest.raises(SystemExit, match="stdio, streamable-http"):
        config.transport()


@pytest.mark.parametrize("port", ["abc", "0", "70000"])
def test_bad_port(monkeypatch, port):
    monkeypatch.setenv("MCP_PORT", port)

    with pytest.raises(SystemExit, match="MCP_PORT"):
        config.port()


def test_http_server_not_running(monkeypatch, free_port):
    monkeypatch.setenv("MCP_PORT", str(free_port))

    async def use():
        async with connect():
            pass

    with pytest.raises(ServerUnavailable, match=f"127.0.0.1:{free_port}/mcp"):
        asyncio.run(use())
