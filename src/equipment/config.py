"""Transport settings, shared by the server and the client."""

import os
from enum import StrEnum


class Transport(StrEnum):
    STDIO = "stdio"  # the client starts the server
    STREAMABLE_HTTP = "streamable-http"  # the server already runs at a URL


def transport() -> Transport:
    value = os.environ.get("MCP_TRANSPORT", Transport.STREAMABLE_HTTP)
    if value not in set(Transport):
        raise SystemExit(f"MCP_TRANSPORT={value!r} must be one of: {', '.join(Transport)}")
    return Transport(value)


def host() -> str:
    """Localhost by default: the server has no login."""
    return os.environ.get("MCP_HOST", "127.0.0.1")


def allowed_hosts() -> list[str]:
    """Host headers the server accepts when not on localhost, e.g. "mcp-server:*"."""
    return [h.strip() for h in os.environ.get("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()]


def port() -> int:
    value = os.environ.get("MCP_PORT", "8000")
    if not value.isdecimal() or not 1 <= int(value) <= 65535:
        raise SystemExit(f"MCP_PORT={value!r} must be a number from 1 to 65535")
    return int(value)


def url() -> str:
    return os.environ.get("MCP_URL") or f"http://{host()}:{port()}/mcp"
