"""The ReAct loop: the model thinks, calls MCP tools, reads the results, and submits a decision.

A reviewer may reject a submitted draft; the issues go back to the model as an observation, so it
can reword the draft or look up what is missing. Safe fallbacks (step limit, refusal, no draft
passing review) are escalated by code, never left to the model.
"""

import json
from collections.abc import Callable
from typing import Any

from anthropic import transform_schema
from anthropic.types import MessageParam, ToolParam, ToolResultBlockParam
from mcp import Client
from pydantic import ValidationError

from equipment.agent.llm import LLMClient
from equipment.agent.mcp_client import anthropic_tools, result_payload
from equipment.agent.prompts import REJECTED, SYSTEM
from equipment.agent.types import (
    AgentResult,
    Draft,
    Request,
    Review,
    Reviewer,
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
FLAG = "flag_for_human_review"
REMINDER = f"Call {SUBMIT} with your final decision."


async def run_agent(
    llm: LLMClient,
    mcp: Client,
    request: Request,
    reviewer: Reviewer | None = None,
    max_steps: int = 8,
    max_revisions: int = 2,
    on_step: Callable[[Step], None] = lambda step: None,
    on_review: Callable[[Review], None] = lambda review: None,
) -> AgentResult:
    """One model turn per step; e.g. E001 "second monitor?" -> approve, WITHIN_LIMIT.

    Without a reviewer the first draft is final.
    """
    steps: list[Step] = []
    reviews: list[Review] = []

    def record(step: Step) -> None:
        steps.append(step)
        on_step(step)

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
            if draft and reviewer:
                review = await reviewer(request, draft, steps)
            issues = review.issues if review else []
            outcome = {"accepted": False, "issues": issues} if issues else {"accepted": True}
            record(Step(thought, SUBMIT, submitted, outcome, bool(issues)))
            if review:
                reviews.append(review)
                on_review(review)
            if draft and not issues:
                return AgentResult(draft, Stop.SUBMITTED, steps, reviews)
            if reviews[-1].no_verdict or len(reviews) > max_revisions:
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
            if index and not thought:
                thought = f"(same turn as step {first_step})"
            arguments = _arguments(request, call)
            observation, is_error = await _run_tool(mcp, request, call.name, arguments, steps)
            record(Step(thought, call.name, arguments, observation, is_error))
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


def _parse(submitted: dict[str, Any]) -> tuple[Draft | None, Review | None]:
    """A valid draft, or a review that rejects it, e.g. for decision "maybe"."""
    try:
        return Draft.model_validate(submitted), None
    except ValidationError as error:
        return None, Review(submitted, validation_issues(error))


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
    if name == SUBMIT:
        return {"error": f"Call {SUBMIT} on its own, after you have read the other results."}, True
    employee_id = arguments.get("employee_id")
    if employee_id is not None and str(employee_id).strip() != request.employee_id:
        return {"error": f"Use the employee ID from the request: {request.employee_id}."}, True
    flagged = saved_escalation(steps)
    if name == FLAG and flagged:
        existing = flagged.observation["escalation_id"]
        return {"error": f"Already escalated as {existing}; submit that escalation_id."}, True
    result = await mcp.call_tool(name, arguments)
    return result_payload(result), bool(result.is_error)


def saved_escalation(steps: list[Step]) -> Step | None:
    """The step that saved an escalation in this run, e.g. ESC-3f9a1c2e, or None."""
    for step in steps:
        saved = isinstance(step.observation, dict) and step.observation.get("escalation_id")
        if step.action == FLAG and saved:
            return step
    return None


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
