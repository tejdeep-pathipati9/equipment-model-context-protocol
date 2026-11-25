"""Scripted model replies, shared by the agent tests."""

from types import SimpleNamespace


class FakeLLM:
    """Returns scripted replies in order and records what it was sent."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = []

    async def create(self, *, system, messages, tools):
        self.calls.append({"system": system, "messages": list(messages), "tools": tools})
        return self.replies.pop(0)


def reply(*blocks, stop="tool_use"):
    return SimpleNamespace(stop_reason=stop, content=list(blocks))


def thought(text):
    return SimpleNamespace(type="text", text=text)


def call(name, arguments, id="c1"):
    return SimpleNamespace(type="tool_use", id=id, name=name, input=arguments)
