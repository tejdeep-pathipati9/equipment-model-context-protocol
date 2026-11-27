"""Capture a paired comparison around the existing agent; no extra model or judge calls."""

from dataclasses import asdict
from time import perf_counter
from typing import Any

from anthropic.types import Message, MessageParam, ToolParam

from equipment.agent.llm import LLMClient
from equipment.agent.prompts import CRITIC
from equipment.agent.reflection import critique
from equipment.agent.types import AgentResult, Draft, Request, Step


class Comparison:
    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm
        self.started = perf_counter()
        self.calls: list[dict[str, Any]] = []
        self.first_draft: dict[str, Any] | None = None
        self.first_steps: list[dict[str, Any]] = []
        self.first_call_count = 0
        self.first_seconds: float | None = None

    async def create(
        self, *, system: str, messages: list[MessageParam], tools: list[ToolParam]
    ) -> Message:
        """Keep SDK usage, including cache fields; failed calls remain visible too."""
        call: dict[str, Any] = {"phase": "critic" if system == CRITIC else "drafter", "usage": None}
        self.calls.append(call)
        started = perf_counter()
        try:
            reply = await self.llm.create(system=system, messages=messages, tools=tools)
            call["usage"] = reply.usage.model_dump(mode="json")
            return reply
        finally:
            call["seconds"] = perf_counter() - started

    async def review(self, request: Request, draft: Draft, steps: list[Step]) -> list[str] | None:
        """run_agent calls this only after schema validation and deterministic checks pass."""
        if self.first_draft is None:
            self.first_draft = draft.model_dump(mode="json")
            self.first_steps = [asdict(step) for step in steps]
            self.first_call_count = len(self.calls)
            self.first_seconds = perf_counter() - self.started
        return await critique(self, request, draft, steps)

    def report(self, case: dict[str, Any], result: AgentResult) -> dict[str, Any]:
        final = result.decision.model_dump(mode="json")
        expected = case["expected"]
        before = matches(self.first_draft, expected)
        after = matches(final, expected)
        return {
            "case": case,
            "first_valid_draft": self.first_draft,
            "first_draft_evidence": self.first_steps,
            "final_decision": final,
            "before_correct": before,
            "after_correct": after,
            "decision_change": {
                (False, False): "still_incorrect",
                (False, True): "corrected",
                (True, False): "regressed",
                (True, True): "stayed_correct",
            }[before, after],
            "baseline_available": self.first_draft is not None,
            "stop": result.stop,
            "recorded": result.recorded,
            "reviews": [asdict(review) for review in result.reviews],
            "steps": [asdict(step) for step in result.steps],
            "calls": self.calls,
            "before_call_count": self.first_call_count,
            "before_usage": usage_totals(self.calls[: self.first_call_count]),
            "total_usage": usage_totals(self.calls),
            "before_seconds": self.first_seconds,
            "total_seconds": perf_counter() - self.started,
            "manual_message_review": {"before_pass": None, "after_pass": None, "notes": ""},
        }


def matches(answer: dict[str, Any] | None, expected: dict[str, Any]) -> bool:
    """Only labeled fields are exact matches; message quality is reviewed by a person."""
    return answer is not None and all(answer.get(key) == value for key, value in expected.items())


def usage_totals(calls: list[dict[str, Any]]) -> dict[str, int]:
    """Cache usage is kept separate from regular input because billing rates can differ."""
    fields = (
        "input_tokens",
        "output_tokens",
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
    )
    return {
        field: sum((call["usage"] or {}).get(field, 0) or 0 for call in calls) for field in fields
    }
