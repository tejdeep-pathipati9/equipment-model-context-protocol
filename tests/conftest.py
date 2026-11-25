"""Shared pytest fixtures."""

import os
import socket
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from datetime import date

import pytest

from equipment.data.seed import create_schema, seed_data
from equipment.data.sqlite_repo import SqliteRepository


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """Fresh, empty temp DB path per test, exported as EQUIP_DB_PATH."""
    path = tmp_path / "equipment.db"
    monkeypatch.setenv("EQUIP_DB_PATH", str(path))
    return path


@pytest.fixture
def conn(db_path):
    """Open connection to a fresh DB with the schema applied."""
    connection = sqlite3.connect(db_path)
    create_schema(connection)
    yield connection
    connection.close()


@pytest.fixture
def seeded_conn(conn):
    """Fresh DB with the schema and the demo seed data."""
    seed_data(conn)
    return conn


@pytest.fixture
def as_of():
    """Fixed 'today' used by the demo and tests (docs/requirements.md 2.2)."""
    return date(2026, 12, 1)


@pytest.fixture
def repo(seeded_conn):
    return SqliteRepository(seeded_conn)


@pytest.fixture
def server_db(db_path, monkeypatch):
    """Seeded DB file the server subprocess can see (EQUIP_DB_PATH, AS_OF_DATE forwarded)."""
    with closing(sqlite3.connect(db_path)) as conn, conn:  # `with conn` commits
        create_schema(conn)
        seed_data(conn)
    monkeypatch.setenv("AS_OF_DATE", "2026-12-01")
    return db_path


@pytest.fixture
def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture(params=["stdio", "streamable-http"])
def mcp_server(request, server_db, free_port, monkeypatch):
    """The server over each transport; with HTTP it runs as its own process."""
    monkeypatch.delenv("MCP_URL", raising=False)
    monkeypatch.delenv("MCP_HOST", raising=False)
    monkeypatch.setenv("MCP_TRANSPORT", request.param)
    monkeypatch.setenv("MCP_PORT", str(free_port))
    if request.param == "stdio":
        yield server_db
        return

    process = subprocess.Popen(
        [sys.executable, "-m", "equipment.server"],
        env=dict(os.environ),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_until_listening(process, free_port)
        yield server_db
    finally:
        process.terminate()
        process.wait(timeout=10)


def wait_until_listening(process, port, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise TimeoutError(f"server not listening on port {port}")
