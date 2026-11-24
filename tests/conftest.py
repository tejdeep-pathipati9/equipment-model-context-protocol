"""Shared pytest fixtures."""

import sqlite3

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
def repo(seeded_conn):
    return SqliteRepository(seeded_conn)
