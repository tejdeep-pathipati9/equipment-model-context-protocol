"""Foundry settings and MCP-to-Anthropic tool conversion. No Azure."""

from types import SimpleNamespace

import pytest

from equipment.agent.llm import FoundryClient
from equipment.agent.mcp_client import anthropic_tools


def test_anthropic_tools_keeps_name_description_and_schema():
    schema = {"type": "object", "properties": {}}
    tool = SimpleNamespace(name="a", description="does a", input_schema=schema)

    assert anthropic_tools([tool]) == [
        {"name": "a", "description": "does a", "input_schema": schema}
    ]


URL = "https://x.services.ai.azure.com/anthropic"


@pytest.fixture
def foundry_env(monkeypatch):
    """Key and model set; resource and base URL left to each test."""
    for name in ("ANTHROPIC_FOUNDRY_RESOURCE", "ANTHROPIC_FOUNDRY_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "key")
    monkeypatch.setenv("AGENT_MODEL", "my-deployment")
    return monkeypatch


def test_foundry_client_names_missing_settings(foundry_env):
    foundry_env.delenv("ANTHROPIC_FOUNDRY_API_KEY")

    expected = "ANTHROPIC_FOUNDRY_API_KEY, ANTHROPIC_FOUNDRY_BASE_URL or ANTHROPIC_FOUNDRY_RESOURCE"
    with pytest.raises(SystemExit, match=expected):
        FoundryClient.from_env()


@pytest.mark.parametrize(
    "setting", [("ANTHROPIC_FOUNDRY_BASE_URL", URL), ("ANTHROPIC_FOUNDRY_RESOURCE", "x")]
)
def test_foundry_client_accepts_base_url_or_resource(foundry_env, setting):
    foundry_env.setenv(*setting)

    assert isinstance(FoundryClient.from_env(), FoundryClient)


def test_foundry_client_rejects_both(foundry_env):
    foundry_env.setenv("ANTHROPIC_FOUNDRY_BASE_URL", URL)
    foundry_env.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "x")

    with pytest.raises(SystemExit, match="not both"):
        FoundryClient.from_env()


def test_foundry_client_points_a_url_resource_to_base_url(foundry_env):
    foundry_env.setenv("ANTHROPIC_FOUNDRY_RESOURCE", URL)

    with pytest.raises(SystemExit, match="put the URL in ANTHROPIC_FOUNDRY_BASE_URL"):
        FoundryClient.from_env()
