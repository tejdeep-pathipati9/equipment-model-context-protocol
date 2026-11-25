"""Choosing how the client and server connect (MCP_TRANSPORT and friends)."""

import asyncio

import pytest
from mcp import StdioServerParameters

from equipment import config
from equipment.agent.mcp_client import ServerUnavailable, connect, connection
from equipment.server import transport_security


@pytest.fixture(autouse=True)
def no_mcp_settings(monkeypatch):
    for name in ("MCP_TRANSPORT", "MCP_HOST", "MCP_PORT", "MCP_URL", "MCP_ALLOWED_HOSTS"):
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


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_localhost_uses_the_sdk_protection(monkeypatch, host):
    monkeypatch.setenv("MCP_HOST", host)

    assert transport_security() is None


def test_other_host_needs_an_allow_list(monkeypatch):
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")

    with pytest.raises(SystemExit, match="MCP_ALLOWED_HOSTS"):
        transport_security()


def test_other_host_checks_the_allow_list(monkeypatch):
    monkeypatch.setenv("MCP_HOST", "0.0.0.0")
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "mcp-server:*, localhost:*")

    security = transport_security()

    assert security.enable_dns_rebinding_protection is True
    assert security.allowed_hosts == ["mcp-server:*", "localhost:*"]
    assert security.allowed_origins == ["http://mcp-server:*", "http://localhost:*"]
