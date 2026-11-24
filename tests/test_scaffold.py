"""Scaffold checks: package imports, seed creates the schema, CLI parses args."""

import importlib

import pytest

from equipment.agent.cli import build_parser
from equipment.data import seed

MODULES = [
    "equipment",
    "equipment.server",
    "equipment.tools",
    "equipment.domain.models",
    "equipment.domain.policy",
    "equipment.data.repository",
    "equipment.data.sqlite_repo",
    "equipment.data.seed",
    "equipment.agent.cli",
    "equipment.agent.llm",
    "equipment.agent.mcp_client",
    "equipment.agent.react",
    "equipment.agent.prompts",
    "equipment.agent.reflection",
    "equipment.agent.trace",
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


def test_cli_parses_scenario():
    args = build_parser().parse_args(["--scenario", "3"])
    assert args.scenario == 3


def test_cli_parses_employee_and_text():
    args = build_parser().parse_args(["--employee", "E004", "--text", "need a laptop"])
    assert (args.employee, args.text) == ("E004", "need a laptop")


def test_cli_requires_a_source():
    with pytest.raises(SystemExit):
        build_parser().parse_args([])
