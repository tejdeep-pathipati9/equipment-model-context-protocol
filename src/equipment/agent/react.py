"""The ReAct loop: the model thinks, calls MCP tools, reads the results, and submits a decision.

Safe fallbacks (step limit, refusal) are escalated by code, never left to the model.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from mcp import Client

from equipment.agent.llm import LLMClient
from equipment.agent.mcp_client import anthropic_tools, result_payload
from equipment.agent.prompts import SYSTEM
from equipment.domain.models import Decision, ReasonCode

SUBMIT = "submit_decision"
SUBMIT_TOOL = {
    "name": SUBMIT,
    "description": "Submit the final decision. Call it once, on its own, after the other tools.",
    "strict": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": list(Decision)},
            "reason_code": {"type": "string", "enum": list(ReasonCode)},
            "item": {"type": ["string", "null"]},
            "message_to_employee": {"type": "string"},
            "escalation_id": {"type": ["string", "null"]},
        },
        "required": ["decision", "reason_code", "item", "message_to_employee", "escalation_id"],
        "additionalProperties": False,
    },
}
REMINDER = f"Call {SUBMIT} with your final decision."


@dataclass(frozen=True)
class Request:
    employee_id: str  # from metadata, never from the text
    text: str


@dataclass(frozen=True)
class Step:
    thought: str
    action: str
    arguments: dict[str, Any]
    observation: Any
    is_error: bool


@dataclass
class AgentResult:
    decision: dict[str, Any]
    stop: str  # "submitted", "step_limit" or "refused"
    steps: list[Step] = field(default_factory=list)


async def run_agent(
    llm: LLMClient,
    mcp: Client,
    request: Request,
    max_steps: int = 8,
    on_step: Callable[[Step], None] = lambda step: None,
) -> AgentResult:
    """One model turn per step; e.g. E001 "second monitor?" -> approve, WITHIN_LIMIT."""
    steps: list[Step] = []

    def record(step: Step) -> None:
        steps.append(step)
        on_step(step)

    tools = [*anthropic_tools((await mcp.list_tools()).tools), SUBMIT_TOOL]
    messages: list[dict[str, Any]] = [
        {"role": "user", "content": f"Employee ID: {request.employee_id}\n\n{request.text}"}
    ]

    for _ in range(max_steps):
        reply = await llm.create(system=SYSTEM, messages=messages, tools=tools)
        if reply.stop_reason == "refusal":
            return await _escalate(mcp, request, ReasonCode.AGENT_REFUSED, "refused", steps, record)

        messages.append({"role": "assistant", "content": reply.content})
        calls = _calls_with_thoughts(reply.content)
        if not calls:
            messages.append({"role": "user", "content": REMINDER})
            continue

        results, first_step = [], len(steps) + 1
        for index, (thought, call) in enumerate(calls):
            if index and not thought:
                thought = f"(same turn as step {first_step})"
            if call.name == SUBMIT and len(calls) == 1:
                record(Step(thought, SUBMIT, dict(call.input), {"accepted": True}, False))
                return AgentResult(dict(call.input), "submitted", steps)
            observation, is_error = await _run_tool(mcp, request, call)
            record(Step(thought, call.name, dict(call.input), observation, is_error))
            results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": json.dumps(observation),
                    "is_error": is_error,
                }
            )
        messages.append({"role": "user", "content": results})

    return await _escalate(mcp, request, ReasonCode.AGENT_STEP_LIMIT, "step_limit", steps, record)


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


async def _run_tool(mcp: Client, request: Request, call: Any) -> tuple:
    if call.name == SUBMIT:
        return {"error": f"Call {SUBMIT} on its own, after you have read the other results."}, True
    employee_id = call.input.get("employee_id")
    if employee_id is not None and str(employee_id).strip() != request.employee_id:
        return {"error": f"Use the employee ID from the request: {request.employee_id}."}, True
    result = await mcp.call_tool(call.name, dict(call.input))
    return result_payload(result), bool(result.is_error)


async def _escalate(mcp, request, reason_code, stop, steps, record) -> AgentResult:
    reason = {
        ReasonCode.AGENT_STEP_LIMIT: "The agent reached its step limit without a decision.",
        ReasonCode.AGENT_REFUSED: "The model declined to handle this request.",
    }[reason_code]
    arguments = {
        "employee_id": request.employee_id,
        "request": request.text,
        "reason": reason,
        "reason_code": reason_code,
    }
    observation = result_payload(await mcp.call_tool("flag_for_human_review", arguments))
    if not isinstance(observation, dict):  # the safe path must not crash on a plain-text result
        observation = {"error": str(observation)}
    record(Step(reason, "flag_for_human_review", arguments, observation, "error" in observation))
    escalation_id = observation.get("escalation_id")
    decision = {
        "decision": Decision.ESCALATE,
        "reason_code": reason_code,
        "item": None,
        "message_to_employee": "Your request could not be decided automatically, so it has "
        f"been passed to a human reviewer (reference {escalation_id}).",
        "escalation_id": escalation_id,
    }
    return AgentResult(decision, stop, steps)
