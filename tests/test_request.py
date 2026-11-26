"""The request is checked once, where it enters: CLI, scenario file or a Python caller."""

import pytest

from equipment.agent import cli
from equipment.agent.types import Request


@pytest.fixture
def started(monkeypatch):
    """The requests the CLI would run, without Foundry or MCP."""
    requests = []

    async def fake_run(request):
        requests.append(request)
        return 0

    monkeypatch.setattr(cli, "run", fake_run)
    return requests


def test_request_trims_the_employee_id():
    assert Request(" E001 ", "Can I get a headset?").employee_id == "E001"


def test_request_keeps_the_employees_wording():
    assert Request("E001", "  my monitor  died ").text == "  my monitor  died "


@pytest.mark.parametrize("employee_id", ["", "   ", None])
def test_request_needs_an_employee_id(employee_id):
    with pytest.raises(ValueError, match="employee ID"):
        Request(employee_id, "Can I get a headset?")


@pytest.mark.parametrize("text", ["", "   ", None])
def test_request_needs_text(text):
    with pytest.raises(ValueError, match="text"):
        Request("E001", text)


def test_cli_rejects_scenario_0(started):
    with pytest.raises(SystemExit, match="1-4"):
        cli.main(["--scenario", "0"])


def test_cli_rejects_blank_text(started):
    with pytest.raises(SystemExit):
        cli.main(["--employee", "E001", "--text", "   "])

    assert started == []


def test_cli_runs_with_a_trimmed_employee_id(started):
    with pytest.raises(SystemExit):
        cli.main(["--employee", " E001 ", "--text", "Can I get a headset?"])

    assert started == [Request("E001", "Can I get a headset?")]
