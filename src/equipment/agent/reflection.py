"""Reflection: the critic, a separate model call that checks a draft (docs/requirements.md 5).

It runs only after the code check (checks.py) passes. The ReAct loop (react.py) decides what
happens to a rejected draft.
"""

import json
from typing import Any

from anthropic import transform_schema
from anthropic.types import Message, ToolParam
from pydantic import ValidationError

from equipment.agent.llm import LLMClient
from equipment.agent.prompts import CRITIC
from equipment.agent.react import SUBMIT
from equipment.agent.types import CriticReview, Draft, Proposal, Request, Step

REVIEW = "submit_review"
REVIEW_TOOL: ToolParam = {
    "name": REVIEW,
    "description": "Submit the review of the draft. An empty list means the draft is correct.",
    "strict": True,
    "input_schema": transform_schema(CriticReview),
}


async def critique(
    llm: LLMClient, request: Request, draft: Draft | Proposal, steps: list[Step]
) -> list[str] | None:
    """The critic's issues, e.g. '"by Friday": ...'. Empty = pass, None = no verdict."""
    key = "draft" if isinstance(draft, Draft) else "proposed_escalation"
    content = json.dumps(_evidence(request, steps) | {key: draft.model_dump(mode="json")})
    reply = await llm.create(
        system=CRITIC, messages=[{"role": "user", "content": content}], tools=[REVIEW_TOOL]
    )
    try:
        review = CriticReview.model_validate(_tool_input(reply, REVIEW))
    except ValidationError:  # also a refusal or a cut-off reply, which have no tool input
        return None
    return [f'"{issue.quote}": {issue.problem}' for issue in review.issues]


def _evidence(request: Request, steps: list[Step]) -> dict[str, Any]:
    """What the draft is judged against. Thoughts are left out so they cannot sway the critic."""
    return {
        "request": {"employee_id": request.employee_id, "text": request.text},
        "tool_results": [
            {"tool": step.action, "arguments": step.arguments, "result": step.observation}
            for step in steps
            if step.action != SUBMIT
        ],
    }


def _tool_input(reply: Message, name: str) -> dict[str, Any] | None:
    """The input of a complete tool call, or None for a refusal, text or cut-off reply."""
    if reply.stop_reason != "tool_use":
        return None
    for block in reply.content:
        if block.type == "tool_use" and block.name == name:
            return dict(block.input)
    return None
