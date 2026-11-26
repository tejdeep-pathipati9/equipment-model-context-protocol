"""The ReAct loop: the model thinks, calls MCP lookup tools, reads results, and submits a decision.

Each submitted draft goes through the code check, then the critic. A rejected draft's issues go back
to the model as an observation, so it can reword the draft or look up what is missing. The model
never writes: once an escalation passes review, code saves it once with flag_for_human_review and
adds the reference to the reply. Safe fallbacks (step limit, refusal, no draft passing review) are
escalated by code, never left to the model.
"""

import json
import re
from collections.abc import Callable
from typing import Any

from anthropic import transform_schema
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam
from mcp import Client
from pydantic import ValidationError

from equipment.agent.checks import check_draft, eligibility_for
from equipment.agent.llm import LLMClient
from equipment.agent.mcp_client import anthropic_tools, result_payload
from equipment.agent.prompts import REJECTED, SYSTEM
from equipment.agent.types import (
    AgentResult,
    Critic,
    Draft,
    FinalDecision,
    Receipt,
    Request,
    Review,
    Step,
    Stop,
    validation_issues,
)
from equipment.domain.models import Decision, ReasonCode

SUBMIT = "submit_decision"
SUBMIT_TOOL: ToolParam = {
    "name": SUBMIT,
    "description": "Submit the final decision. Call it once, on its own, after the other tools.",
    "strict": True,
    "input_schema": transform_schema(Draft),
}
FLAG = "flag_for_human_review"  # the one tool that writes: called by code, never offered
REMINDER = f"Call {SUBMIT} with your final decision."
THOUGHT_LABEL = re.compile(r"^\s*Thought:", re.MULTILINE)

HANDED_OFF = "Sent for human review. Reference: {escalation_id}."
UNDECIDED = "Your request could not be decided automatically."
NOT_CONFIRMED = (
    "Your request could not be confirmed as sent for human review. Please submit it again."
)


async def run_agent(
    llm: LLMClient,
    mcp: Client,
    request: Request,
    critic: Critic,
    max_steps: int = 8,
    max_revisions: int = 2,
    on_step: Callable[[Step], None] = lambda step: None,
    on_review: Callable[[Review], None] = lambda review: None,
) -> AgentResult:
    """One model turn per step; e.g. E001 "second monitor?" -> approve, WITHIN_LIMIT."""
    steps: list[Step] = []
    reviews: list[Review] = []

    def record(step: Step) -> None:
        steps.append(step)
        on_step(step)

    def keep(review: Review) -> bool:
        """Keep a review; True when the run must stop (no verdict, or too many rejections)."""
        reviews.append(review)
        on_review(review)
        rejections = sum(bool(kept.issues) for kept in reviews)
        return bool(review.issues) and (review.no_verdict or rejections > max_revisions)

    async def give_up(stop: Stop, reason_code: ReasonCode, reason: str) -> AgentResult:
        decision = await escalate(mcp, request, reason_code, reason, record)
        return AgentResult(decision, stop, steps, reviews)

    offered = [
        tool for tool in anthropic_tools((await mcp.list_tools()).tools) if tool["name"] != FLAG
    ]
    tools = [*offered, SUBMIT_TOOL]
    messages: list[MessageParam] = [
        {"role": "user", "content": f"Employee ID: {request.employee_id}\n\n{request.text}"}
    ]

    for _ in range(max_steps):
        reply = await llm.create(system=SYSTEM, messages=messages, tools=tools)
        if reply.stop_reason == "refusal":
            return await give_up(
                Stop.REFUSED, ReasonCode.AGENT_REFUSED, "The model declined to handle this request."
            )

        messages.append({"role": "assistant", "content": reply.content})
        calls = _calls_with_thoughts(reply.content)
        if not calls:
            messages.append({"role": "user", "content": REMINDER})
            continue

        if len(calls) == 1 and calls[0][1].name == SUBMIT:
            [(thought, call)] = calls
            submitted = dict(call.input)
            draft, review = _parse(submitted)
            if draft:
                review = await _review(critic, request, draft, steps)
            issues = review.issues
            outcome = {"accepted": False, "issues": issues} if issues else {"accepted": True}
            record(Step(thought, SUBMIT, submitted, outcome, bool(issues)))
            must_stop = keep(review)
            if draft and not issues:
                decision = await finish(mcp, request, draft, record)
                return AgentResult(decision, Stop.SUBMITTED, steps, reviews)
            if draft and review.no_verdict and draft.decision == Decision.ESCALATE:
                # It passed the code check; keep its reason code, but say it was not reviewed.
                return await give_up(
                    Stop.REFLECTION_FAILED, draft.reason_code, _unreviewed(draft, steps)
                )
            if must_stop:
                return await give_up(
                    Stop.REFLECTION_FAILED,
                    ReasonCode.REFLECTION_FAILED,
                    f"The draft did not pass review: {'; '.join(issues)}",
                )
            messages.append({"role": "user", "content": [_rejected(call.id, outcome)]})
            continue

        results: list[ToolResultBlockParam] = []
        first_step = len(steps) + 1
        for index, (thought, call) in enumerate(calls):
            # A later call in the same reply usually has no thought of its own.
            shared = first_step if index and not thought else None
            arguments = dict(call.input)
            observation, is_error = await _run_tool(mcp, request, call.name, arguments)
            record(Step(thought, call.name, arguments, observation, is_error, same_turn_as=shared))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(observation),
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})

    return await give_up(
        Stop.STEP_LIMIT,
        ReasonCode.AGENT_STEP_LIMIT,
        "The agent reached its step limit without a decision.",
    )


def _parse(submitted: dict[str, Any]) -> tuple[Draft | None, Review]:
    """A valid draft, or a review that rejects it, e.g. for decision "maybe"."""
    try:
        return Draft.model_validate(submitted), Review(submitted, [])
    except ValidationError as error:
        return None, Review(submitted, validation_issues(error))


async def _review(critic: Critic, request: Request, draft: Draft, steps: list[Step]) -> Review:
    """The code check, then the critic only if it passes."""
    submitted = draft.model_dump(mode="json")
    code_issues = check_draft(draft, steps)
    if code_issues:
        return Review(submitted, code_issues)
    return Review(submitted, [], await critic(request, draft, steps))


def _unreviewed(draft: Draft, steps: list[Step]) -> str:
    """The reviewer's note when the critic gave no verdict: what was verified, what was not."""
    result = eligibility_for(draft.item, steps)
    verified = (
        f"eligibility for {result.item} is {result.status} ({result.reason_code})"
        if result
        else "no eligibility result"
    )
    # The model's own note is left out: nobody reviewed it. The request is saved separately.
    return (
        f"Proposed {draft.reason_code}. Verified by code: {verified}. "
        "Not reviewed: the critic gave no verdict."
    )


def _rejected(tool_use_id: str, rejection: dict[str, Any]) -> ToolResultBlockParam:
    """The issues, and how to fix them."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [
            {"type": "text", "text": json.dumps(rejection)},
            {"type": "text", "text": REJECTED},
        ],
        "is_error": True,
    }


def _calls_with_thoughts(content: list[Any]) -> list[tuple[str, Any]]:
    """Each tool call paired with the text the model wrote just before it."""
    calls, text = [], []
    for block in content:
        if block.type == "text":  # a "Thought:" label may start any line of any block
            text.append(" ".join(THOUGHT_LABEL.sub("", block.text).split()))
        elif block.type == "tool_use":
            calls.append((" ".join(part for part in text if part), block))
            text = []
    return calls


async def _run_tool(
    mcp: Client, request: Request, name: str, arguments: dict[str, Any]
) -> tuple[Any, bool]:
    blocked = _blocked(request, name, arguments)
    if blocked:
        return {"error": blocked}, True
    result = await mcp.call_tool(name, arguments)
    return result_payload(result), bool(result.is_error)


def _blocked(request: Request, name: str, arguments: dict[str, Any]) -> str:
    """Why code refuses this call before it reaches the server, or "" if it may run."""
    if name == SUBMIT:
        return f"Call {SUBMIT} on its own, after you have read the other results."
    if name == FLAG:
        return f"Submit an escalation with {SUBMIT}; the system saves it after review."
    employee_id = arguments.get("employee_id")
    if employee_id is not None and str(employee_id).strip() != request.employee_id:
        return f"Use the employee ID from the request: {request.employee_id}."
    return ""


async def finish(
    mcp: Client, request: Request, draft: Draft, record: Callable[[Step], None]
) -> FinalDecision:
    """The accepted draft as the employee gets it; an escalation is saved here, once."""
    if draft.decision != Decision.ESCALATE:
        return FinalDecision.from_draft(draft)
    reason = draft.escalation_reason or ""  # always set: Draft requires it for an escalation
    thought = "The escalation passed review; the system saves it once."
    escalation_id = await _save(mcp, request, draft.reason_code, reason, thought, record)
    if escalation_id is None:
        return FinalDecision.from_draft(draft, message=NOT_CONFIRMED)
    handed_off = HANDED_OFF.format(escalation_id=escalation_id)
    return FinalDecision.from_draft(
        draft, escalation_id, f"{draft.message_to_employee} {handed_off}"
    )


def _receipt(observation: dict[str, Any]) -> str | None:
    """The escalation ID, only from a complete receipt for an open escalation."""
    try:
        return Receipt.model_validate(observation).escalation_id
    except ValidationError:
        return None


async def escalate(
    mcp: Client,
    request: Request,
    reason_code: ReasonCode,
    reason: str,
    record: Callable[[Step], None],
) -> FinalDecision:
    """The safe decision, made by code: escalate with a standard message."""
    escalation_id = await _save(mcp, request, reason_code, reason, reason, record)
    if escalation_id:
        message = f"{UNDECIDED} {HANDED_OFF.format(escalation_id=escalation_id)}"
    else:
        message = NOT_CONFIRMED
    return FinalDecision(
        decision=Decision.ESCALATE,
        reason_code=reason_code,
        item=None,
        message_to_employee=message,
        escalation_id=escalation_id,
    )


async def _save(
    mcp: Client,
    request: Request,
    reason_code: ReasonCode,
    reason: str,
    thought: str,
    record: Callable[[Step], None],
) -> str | None:
    """Save the escalation; the reviewer always gets the employee's own words."""
    arguments = {
        "employee_id": request.employee_id,
        "request": request.text,
        "reason": reason,
        "reason_code": reason_code,
    }
    try:
        result = await mcp.call_tool(FLAG, arguments)
        observation, failed = result_payload(result), bool(result.is_error)
    except Exception as error:  # e.g. a timeout: it may have been saved, so it is unconfirmed
        observation, failed = (
            {"error": f"the save could not be confirmed: {type(error).__name__}"},
            True,
        )
    if not isinstance(observation, dict):  # the safe path must not crash on a plain-text result
        observation = {"error": str(observation)}
    escalation_id = None if failed else _receipt(observation)
    record(Step(thought, FLAG, arguments, observation, escalation_id is None, actor="code"))
    return escalation_id
