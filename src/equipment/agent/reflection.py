"""Reflection: check the draft decision before it becomes final (docs/requirements.md 5).

Code check, then critic; a failing draft is revised and checked again. If it still fails, it is
escalated by code, so an unreviewed draft never reaches the employee.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from anthropic.types import Message, ToolParam
from mcp import Client

from equipment.agent.llm import LLMClient
from equipment.agent.prompts import CRITIC, REVISE
from equipment.agent.react import (
    SUBMIT,
    SUBMIT_TOOL,
    AgentResult,
    Request,
    Step,
    escalate,
    saved_escalation,
)
from equipment.domain.models import Basis, Decision, ReasonCode, Status

Draft = dict[str, Any]  # the input of submit_decision

MAX_REVISIONS = 2
ELIGIBILITY = "check_request_eligibility"
VERDICTS = {Decision.APPROVE: Status.ELIGIBLE, Decision.DENY: Status.NOT_ELIGIBLE}

REVIEW = "submit_review"
REVIEW_TOOL: ToolParam = {
    "name": REVIEW,
    "description": "Submit the review of the draft. An empty list means the draft is correct.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "issues": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "quote": {"type": "string", "description": "exact words from the draft"},
                        "problem": {"type": "string"},
                    },
                    "required": ["quote", "problem"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["issues"],
        "additionalProperties": False,
    },
}


class Outcome(StrEnum):
    CONFIRMED = "confirmed"  # the first draft passed
    REVISED = "revised"  # a later draft passed
    ESCALATED = "escalated"  # no draft passed
    SKIPPED = "skipped"  # the decision was made by code, not drafted


@dataclass(frozen=True)
class Round:
    """The checks of one draft. The critic only runs once the code check passes."""

    draft: Draft
    code_issues: list[str]
    critic_issues: list[str] | None = None  # None = not run, or no verdict

    @property
    def passed(self) -> bool:
        return not self.code_issues and self.critic_issues == []

    @property
    def no_verdict(self) -> bool:
        return not self.code_issues and self.critic_issues is None

    @property
    def issues(self) -> list[str]:
        return self.code_issues or self.critic_issues or []


@dataclass(frozen=True)
class Reflection:
    decision: Draft
    outcome: Outcome
    rounds: list[Round] = field(default_factory=list)
    reason: str | None = None  # why it was escalated


async def reflect(
    llm: LLMClient,
    mcp: Client,
    request: Request,
    result: AgentResult,
    max_revisions: int = MAX_REVISIONS,
    on_round: Callable[[Round], None] = lambda round: None,
) -> Reflection:
    """Check, revise, check again; e.g. "by Friday" is caught, removed, then confirmed."""
    if result.stop != "submitted":
        return Reflection(result.decision, Outcome.SKIPPED)

    draft, rounds = result.decision, []
    while True:
        current = await _review(llm, request, draft, result.steps)
        rounds.append(current)
        on_round(current)
        if current.passed:
            outcome = Outcome.CONFIRMED if len(rounds) == 1 else Outcome.REVISED
            return Reflection(draft, outcome, rounds)
        if current.no_verdict:
            reason = "The critic gave no verdict on the draft."
            break
        reason = f"The draft did not pass review: {'; '.join(current.issues)}"
        if len(rounds) > max_revisions:
            break
        revised = await revise(llm, request, draft, result.steps, current.issues)
        if revised is None or _switches_escalation(draft, revised):
            break
        draft = revised

    decision = await escalate(
        mcp, request, ReasonCode.REFLECTION_FAILED, reason, result.steps, result.steps.append
    )
    return Reflection(decision, Outcome.ESCALATED, rounds, reason)


async def _review(llm: LLMClient, request: Request, draft: Draft, steps: list[Step]) -> Round:
    code_issues = check_draft(draft, steps)
    if code_issues:
        return Round(draft, code_issues)
    return Round(draft, [], await critique(llm, request, draft, steps))


def _switches_escalation(before: Draft, after: Draft) -> bool:
    """Only code may start an escalation here, and nothing may undo one."""
    return (before["decision"] == Decision.ESCALATE) != (after["decision"] == Decision.ESCALATE)


# ---------------------------------------------------------------- code check


def check_draft(draft: Draft, steps: list[Step]) -> list[str]:
    """Problems code can prove, e.g. approve when the tool said not_eligible. Empty = pass."""
    if draft["decision"] == Decision.ESCALATE:
        return _check_escalation(draft, steps)
    return _check_verdict(draft, steps)


def _check_verdict(draft: Draft, steps: list[Step]) -> list[str]:
    """Approve or deny must match the last eligibility result."""
    decision, code = draft["decision"], draft["reason_code"]
    flagged = saved_escalation(steps)
    if flagged:
        saved = flagged.observation["escalation_id"]
        return [f"{decision}, but this request was already escalated as {saved}"]

    result = _last_eligibility(steps)
    if result is None:
        return [f"{decision}, but there is no eligibility result"]

    issues = []
    if draft["escalation_id"] is not None:
        issues.append(f"{decision}, but escalation_id is {draft['escalation_id']}")
    if draft["item"] != result["item"]:
        issues.append(f"item is {draft['item']}, but eligibility was checked for {result['item']}")
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
    if draft["escalation_id"] != saved_id:
        issues.append(f"escalation_id is {draft['escalation_id']}, but {saved_id} was saved")
    if draft["reason_code"] != saved_code:
        issues.append(f"reason code is {draft['reason_code']}, but {saved_code} was saved")
    if saved_id not in draft["message_to_employee"]:
        issues.append(f"the message does not mention {saved_id}")
    return issues


def _last_eligibility(steps: list[Step]) -> dict[str, Any] | None:
    for step in reversed(steps):
        if step.action != ELIGIBILITY or step.is_error:
            continue
        if isinstance(step.observation, dict) and "status" in step.observation:
            return step.observation
    return None


# ---------------------------------------------------------------- critic and reviser


async def critique(
    llm: LLMClient, request: Request, draft: Draft, steps: list[Step]
) -> list[str] | None:
    """The critic's issues, e.g. '"by Friday": ...'. Empty = pass, None = no verdict."""
    content = json.dumps(_evidence(request, steps) | {"draft": draft})
    reply = await llm.create(
        system=CRITIC, messages=[{"role": "user", "content": content}], tools=[REVIEW_TOOL]
    )
    review = _tool_input(reply, REVIEW)
    if review is None:
        return None
    return [f'"{issue["quote"]}": {issue["problem"]}' for issue in review["issues"]]


async def revise(
    llm: LLMClient, request: Request, draft: Draft, steps: list[Step], issues: list[str]
) -> Draft | None:
    """A corrected draft, or None. Has no MCP tools, so it cannot escalate twice."""
    content = json.dumps(_evidence(request, steps) | {"draft": draft, "issues": issues})
    reply = await llm.create(
        system=REVISE, messages=[{"role": "user", "content": content}], tools=[SUBMIT_TOOL]
    )
    return _tool_input(reply, SUBMIT)


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
