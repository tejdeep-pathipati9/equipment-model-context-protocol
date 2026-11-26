"""Scaffold checks: package imports, seed creates the schema, CLI parses args."""

import importlib
import sqlite3
from contextlib import closing

import pytest

from equipment import server
from equipment.agent.cli import build_parser, load_scenario
from equipment.agent.types import Request
from equipment.data import seed

MODULES = [
    "equipment",
    "equipment.server",
    "equipment.tools",
    "equipment.domain.models",
    "equipment.domain.policy",
    "equipment.data.sqlite_repo",
    "equipment.data.seed",
    "equipment.agent.checks",
    "equipment.agent.cli",
    "equipment.agent.llm",
    "equipment.agent.mcp_client",
    "equipment.agent.react",
    "equipment.agent.prompts",
    "equipment.agent.reflection",
    "equipment.agent.trace",
    "equipment.agent.types",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def test_schema_creates_all_tables(conn):
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"employees", "equipment", "policies", "escalations"} <= tables


def test_seed_main_creates_db_once(db_path, capsys):
    seed.main()
    assert db_path.exists()
    assert "created" in capsys.readouterr().out

    seed.main()
    assert "already exists" in capsys.readouterr().out


def old_database(path):
    """A database made before the schema had a version (and its CHECK constraints)."""
    with closing(sqlite3.connect(path)) as conn, conn:
        conn.execute("CREATE TABLE employees (id TEXT PRIMARY KEY)")


def test_new_database_has_the_current_schema_version(conn):
    assert conn.execute("PRAGMA user_version").fetchone()[0] == seed.SCHEMA_VERSION


def test_server_refuses_a_database_from_an_older_schema(db_path):
    old_database(db_path)

    with pytest.raises(RuntimeError, match="older schema"), server.repository():
        pass


def test_seed_refuses_an_older_database(db_path):
    old_database(db_path)

    with pytest.raises(SystemExit, match="older schema"):
        seed.main()


@pytest.fixture
def interrupted_seed(db_path, monkeypatch):
    """A first seed that stops after the schema is created, before the rows are written."""

    def stop(conn):
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr(seed, "seed_data", stop)
        with pytest.raises(KeyboardInterrupt):
            seed.main()


def test_database_appears_only_when_complete(db_path, monkeypatch):
    """A process killed mid-build (no cleanup runs) must not leave a half-built database."""
    exists_while_building = []
    write_rows = seed.seed_data

    def watch(conn):
        exists_while_building.append(db_path.exists())
        write_rows(conn)

    monkeypatch.setattr(seed, "seed_data", watch)

    seed.main()

    assert exists_while_building == [False]


def test_interrupted_seed_leaves_no_database(interrupted_seed, db_path):
    assert not db_path.exists()


def test_seed_after_an_interruption_creates_every_employee(interrupted_seed, db_path):
    seed.main()

    with closing(sqlite3.connect(db_path)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM employees").fetchone()[0] == len(seed.EMPLOYEES)


def test_cli_parses_scenario():
    args = build_parser().parse_args(["--scenario", "3"])
    assert args.scenario == 3


def test_cli_parses_employee_and_text():
    args = build_parser().parse_args(["--employee", "E004", "--text", "need a laptop"])
    assert (args.employee, args.text) == ("E004", "need a laptop")


def test_cli_requires_a_source():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_load_scenario():
    assert load_scenario(3) == Request("E003", "One of my monitors died, can I get a replacement?")


def test_load_scenario_out_of_range():
    with pytest.raises(SystemExit, match="1-4"):
        load_scenario(9)
