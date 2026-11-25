"""Reflection: review a draft decision before it becomes final (docs/requirements.md 5).

A code check, then a critic. The ReAct loop (react.py) decides what happens to a rejected draft.
"""

import json
from typing import Any

from anthropic import transform_schema
from anthropic.types import Message, ToolParam
from pydantic import ValidationError

from equipment.agent.llm import LLMClient
from equipment.agent.prompts import CRITIC
from equipment.agent.react import SUBMIT, saved_escalation
from equipment.agent.types import CriticReview, Draft, Request, Review, Step
from equipment.domain.models import Basis, Decision, ReasonCode, Status

ELIGIBILITY = "check_request_eligibility"
VERDICTS = {Decision.APPROVE: Status.ELIGIBLE, Decision.DENY: Status.NOT_ELIGIBLE}

REVIEW = "submit_review"
REVIEW_TOOL: ToolParam = {
    "name": REVIEW,
    "description": "Submit the review of the draft. An empty list means the draft is correct.",
    "strict": True,
    "input_schema": transform_schema(CriticReview),
}


async def review_draft(llm: LLMClient, request: Request, draft: Draft, steps: list[Step]) -> Review:
    """Both checks of one draft, e.g. code check passes, critic finds "by Friday"."""
    submitted = draft.model_dump(mode="json")
    code_issues = check_draft(draft, steps)
    if code_issues:
        return Review(submitted, code_issues)
    return Review(submitted, [], await critique(llm, request, draft, steps))


# ---------------------------------------------------------------- code check


def check_draft(draft: Draft, steps: list[Step]) -> list[str]:
    """Problems code can prove, e.g. approve when the tool said not_eligible. Empty = pass."""
    if draft.decision == Decision.ESCALATE:
        return _check_escalation(draft, steps)
    return _check_verdict(draft, steps)


def _check_verdict(draft: Draft, steps: list[Step]) -> list[str]:
    """Approve or deny must match the last eligibility result."""
    decision, code = draft.decision, draft.reason_code
    flagged = saved_escalation(steps)
    if flagged:
        saved = flagged.observation["escalation_id"]
        return [f"{decision}, but this request was already escalated as {saved}"]

    result = _last_eligibility(steps)
    if result is None:
        return [f"{decision}, but there is no eligibility result"]

    issues = []
    if draft.escalation_id is not None:
        issues.append(f"{decision}, but escalation_id is {draft.escalation_id}")
    if draft.item != result["item"]:
        issues.append(f"item is {draft.item}, but eligibility was checked for {result['item']}")
    if decision == Decision.DENY and code == ReasonCode.AT_MAX_QUANTITY:
        if result["basis"] != Basis.REFRESH:
            issues.append(f"{code} needs basis refresh, but the basis is {result['basis']}")
    elif result["status"] != VERDICTS[decision]:
        issues.append(f"{decision}, but eligibility is {result['status']}")
    elif code != result["reason_code"]:
        issues.append(f"reason code is {code}, but eligibility says {result['reason_code']}")
    return issues


def _check_escalation(draft: Draft, steps: list[Step]) -> list[str]:
    """An escalation must match the saved record and tell the employee its ID."""
    flagged = saved_escalation(steps)
    if flagged is None:
        return ["escalate, but no escalation was saved"]

    saved_id, saved_code = flagged.observation["escalation_id"], flagged.arguments["reason_code"]
    issues = []
    if draft.escalation_id != saved_id:
        issues.append(f"escalation_id is {draft.escalation_id}, but {saved_id} was saved")
    if draft.reason_code != saved_code:
        issues.append(f"reason code is {draft.reason_code}, but {saved_code} was saved")
    if saved_id not in draft.message_to_employee:
        issues.append(f"the message does not mention {saved_id}")
    return issues


def _last_eligibility(steps: list[Step]) -> dict[str, Any] | None:
    for step in reversed(steps):
        if step.action != ELIGIBILITY or step.is_error:
            continue
        if isinstance(step.observation, dict) and "status" in step.observation:
            return step.observation
    return None


# ---------------------------------------------------------------- critic


async def critique(
    llm: LLMClient, request: Request, draft: Draft, steps: list[Step]
) -> list[str] | None:
    """The critic's issues, e.g. '"by Friday": ...'. Empty = pass, None = no verdict."""
    content = json.dumps(_evidence(request, steps) | {"draft": draft.model_dump(mode="json")})
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
