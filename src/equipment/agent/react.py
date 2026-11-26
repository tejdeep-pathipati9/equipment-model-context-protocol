"""The ReAct loop: the model thinks, calls MCP tools, reads the results, and submits a decision.

Each submitted draft, and each escalation before it is saved, goes through the code check, then the
critic. Rejected issues go back to the model as an observation, so it can reword the draft or look
up what is missing. Safe fallbacks (step limit, refusal, no draft passing review) are escalated by
code, never left to the model.
"""

import json
from collections.abc import Callable
from typing import Any

from anthropic import transform_schema
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam
from mcp import Client
from pydantic import ValidationError

from equipment.agent.checks import FLAG, check_draft, check_escalation, saved_escalation
from equipment.agent.llm import LLMClient
from equipment.agent.mcp_client import anthropic_tools, result_payload
from equipment.agent.prompts import NOT_ESCALATED, REJECTED, SYSTEM
from equipment.agent.types import (
    AgentResult,
    Critic,
    Draft,
    Proposal,
    Request,
    Review,
    Step,
    Stop,
    validation_issues,
)
from equipment.domain.models import ESCALATION_CODES, Decision, ReasonCode

SUBMIT = "submit_decision"
SUBMIT_TOOL: ToolParam = {
    "name": SUBMIT,
    "description": "Submit the final decision. Call it once, on its own, after the other tools.",
    "strict": True,
    "input_schema": transform_schema(Draft),
}
REMINDER = f"Call {SUBMIT} with your final decision."


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
        rejections = sum(bool(kept.issues) and not kept.no_verdict for kept in reviews)
        return bool(review.issues) and (review.no_verdict or rejections > max_revisions)

    async def give_up(stop: Stop, reason_code: ReasonCode, reason: str) -> AgentResult:
        decision = await escalate(mcp, request, reason_code, reason, steps, record)
        return AgentResult(decision, stop, steps, reviews)

    tools = [*anthropic_tools((await mcp.list_tools()).tools), SUBMIT_TOOL]
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
            if keep(review):
                return await give_up(
                    Stop.REFLECTION_FAILED,
                    ReasonCode.REFLECTION_FAILED,
                    f"The draft did not pass review: {'; '.join(issues)}",
                )
            if draft and not issues:
                return AgentResult(draft, Stop.SUBMITTED, steps, reviews)
            messages.append({"role": "user", "content": [_rejected(call.id, outcome)]})
            continue

        results: list[ToolResultBlockParam] = []
        first_step = len(steps) + 1
        for index, (thought, call) in enumerate(calls):
            if index and not thought:
                thought = f"(same turn as step {first_step})"
            arguments = _arguments(request, call)
            review = await _review_escalation(critic, request, call.name, arguments, steps)
            # No verdict on an escalation that passed the code check: save it anyway, since a
            # human will see it, rather than lose its reason code to REFLECTION_FAILED.
            if review and review.issues and not review.no_verdict:
                observation = {"accepted": False, "issues": review.issues}
                record(Step(thought, call.name, arguments, observation, True))
                if keep(review):
                    return await give_up(
                        Stop.REFLECTION_FAILED,
                        ReasonCode.REFLECTION_FAILED,
                        f"The escalation did not pass review: {'; '.join(review.issues)}",
                    )
                results.append(_rejected(call.id, observation, NOT_ESCALATED))
                continue
            observation, is_error = await _run_tool(mcp, request, call.name, arguments, steps)
            record(Step(thought, call.name, arguments, observation, is_error))
            if review:
                keep(review)
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


async def _review_escalation(
    critic: Critic, request: Request, name: str, arguments: dict[str, Any], steps: list[Step]
) -> Review | None:
    """Review an escalation before it is saved; None when the call is not one to review.

    A blocked call or invalid arguments are left to _run_tool and the tool, which save nothing.
    """
    if name != FLAG or _blocked(request, name, arguments, steps):
        return None
    try:
        proposal = Proposal.model_validate(
            {"reason_code": arguments.get("reason_code"), "reason": arguments.get("reason")}
        )
    except ValidationError:
        return None
    if proposal.reason_code not in ESCALATION_CODES:
        return None
    proposed = {"decision": Decision.ESCALATE.value, **proposal.model_dump(mode="json")}
    code_issues = check_escalation(proposal, steps)
    if code_issues:
        return Review(proposed, code_issues)
    return Review(proposed, [], await critic(request, proposal, steps))


def _rejected(
    tool_use_id: str, rejection: dict[str, Any], rules: str = REJECTED
) -> ToolResultBlockParam:
    """The issues, and how to fix them."""
    return {
        "type": "tool_result",
        "tool_use_id": tool_use_id,
        "content": [
            {"type": "text", "text": json.dumps(rejection)},
            {"type": "text", "text": rules},
        ],
        "is_error": True,
    }


def _calls_with_thoughts(content: list[Any]) -> list[tuple[str, Any]]:
    """Each tool call paired with the text the model wrote just before it."""
    calls, text = [], []
    for block in content:
        if block.type == "text":
            text.append(block.text)
        elif block.type == "tool_use":
            thought = " ".join(text).strip().removeprefix("Thought:").strip()
            calls.append((thought, block))
            text = []
    return calls


def _arguments(request: Request, call: Any) -> dict[str, Any]:
    """The model's arguments, except that a reviewer always gets the employee's own words."""
    arguments = dict(call.input)
    if call.name == FLAG:
        arguments["request"] = request.text
    return arguments


async def _run_tool(
    mcp: Client, request: Request, name: str, arguments: dict[str, Any], steps: list[Step]
) -> tuple[Any, bool]:
    blocked = _blocked(request, name, arguments, steps)
    if blocked:
        return {"error": blocked}, True
    result = await mcp.call_tool(name, arguments)
    return result_payload(result), bool(result.is_error)


def _blocked(request: Request, name: str, arguments: dict[str, Any], steps: list[Step]) -> str:
    """Why code refuses this call before it reaches the server, or "" if it may run."""
    if name == SUBMIT:
        return f"Call {SUBMIT} on its own, after you have read the other results."
    employee_id = arguments.get("employee_id")
    if employee_id is not None and str(employee_id).strip() != request.employee_id:
        return f"Use the employee ID from the request: {request.employee_id}."
    flagged = saved_escalation(steps)
    if name == FLAG and flagged:
        existing = flagged.observation["escalation_id"]
        return f"Already escalated as {existing}; submit that escalation_id."
    return ""


async def escalate(
    mcp: Client,
    request: Request,
    reason_code: ReasonCode,
    reason: str,
    steps: list[Step],
    record: Callable[[Step], None],
) -> Draft:
    """The safe decision, made by code. Keeps an escalation already saved in this run, so the
    reply always matches the one record the reviewer sees."""
    flagged = saved_escalation(steps)
    if flagged:
        reason_code = ReasonCode(flagged.arguments["reason_code"])
        escalation_id = flagged.observation["escalation_id"]
    else:
        escalation_id = await _flag(mcp, request, reason_code, reason, record)
    if escalation_id:
        message = (
            "Your request could not be decided automatically, so it has been passed to a human "
            f"reviewer (reference {escalation_id})."
        )
    else:
        message = "Your request could not be decided or recorded. Please submit it again."
    return Draft(
        decision=Decision.ESCALATE,
        reason_code=reason_code,
        item=None,
        message_to_employee=message,
        escalation_id=escalation_id,
    )


async def _flag(
    mcp: Client,
    request: Request,
    reason_code: ReasonCode,
    reason: str,
    record: Callable[[Step], None],
) -> str | None:
    arguments = {
        "employee_id": request.employee_id,
        "request": request.text,
        "reason": reason,
        "reason_code": reason_code,
    }
    observation = result_payload(await mcp.call_tool(FLAG, arguments))
    if not isinstance(observation, dict):  # the safe path must not crash on a plain-text result
        observation = {"error": str(observation)}
    record(Step(reason, FLAG, arguments, observation, "error" in observation))
    return observation.get("escalation_id")
