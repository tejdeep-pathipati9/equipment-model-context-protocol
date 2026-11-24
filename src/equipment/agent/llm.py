"""LLM access: the LLMClient protocol and its Azure Foundry implementation."""

import os
from typing import Protocol

from anthropic import AsyncAnthropicFoundry
from anthropic.types import Message, MessageParam, ToolParam

REQUIRED_ENV = ("ANTHROPIC_FOUNDRY_API_KEY", "ANTHROPIC_FOUNDRY_RESOURCE", "AGENT_MODEL")
MAX_TOKENS = 16000


class LLMClient(Protocol):
    """Lets tests use a fake instead of Foundry."""

    async def create(
        self, *, system: str, messages: list[MessageParam], tools: list[ToolParam]
    ) -> Message: ...


class FoundryClient:
    """No temperature: current Claude models reject sampling settings."""

    def __init__(self, client: AsyncAnthropicFoundry, model: str) -> None:
        self._client = client
        self._model = model

    @classmethod
    def from_env(cls) -> "FoundryClient":
        missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
        if missing:
            raise SystemExit(f"missing settings: {', '.join(missing)} (see .env.example)")
        if "://" in os.environ["ANTHROPIC_FOUNDRY_RESOURCE"]:
            raise SystemExit("ANTHROPIC_FOUNDRY_RESOURCE must be the resource name, not a URL")
        # The SDK reads the key and resource from the environment itself.
        return cls(AsyncAnthropicFoundry(), os.environ["AGENT_MODEL"])

    async def create(
        self, *, system: str, messages: list[MessageParam], tools: list[ToolParam]
    ) -> Message:
        return await self._client.messages.create(
            model=self._model,
            max_tokens=MAX_TOKENS,
            system=system,
            messages=messages,
            tools=tools,
        )
