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


def test_foundry_client_names_missing_settings(monkeypatch):
    for name in ("ANTHROPIC_FOUNDRY_API_KEY", "ANTHROPIC_FOUNDRY_RESOURCE"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AGENT_MODEL", "my-deployment")

    with pytest.raises(SystemExit, match="ANTHROPIC_FOUNDRY_API_KEY, ANTHROPIC_FOUNDRY_RESOURCE"):
        FoundryClient.from_env()


def test_foundry_client_rejects_a_url_as_resource(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_API_KEY", "key")
    monkeypatch.setenv("AGENT_MODEL", "my-deployment")
    monkeypatch.setenv("ANTHROPIC_FOUNDRY_RESOURCE", "https://x.services.ai.azure.com/anthropic")

    with pytest.raises(SystemExit, match="resource name, not a URL"):
        FoundryClient.from_env()
