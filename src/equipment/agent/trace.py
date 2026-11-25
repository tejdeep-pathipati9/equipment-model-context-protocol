"""Thought / Action / Observation printer, compact enough to screenshot."""

import json
from typing import Any

from equipment.agent.react import Step


def format_step(number: int, step: Step) -> str:
    observation = json.dumps(step.observation)
    if step.is_error:
        observation = f"ERROR {observation}"
    return (
        f"Step {number}\n"
        f"  Thought:     {step.thought or '(none)'}\n"
        f"  Action:      {step.action}({json.dumps(step.arguments)})\n"
        f"  Observation: {observation}"
    )


def format_decision(decision: dict[str, Any]) -> str:
    rule = "=" * 72
    lines = [
        rule,
        f"Decision:    {str(decision['decision']).upper()} ({decision['reason_code']})",
        f"Item:        {decision['item'] or '-'}",
        f"Escalation:  {decision['escalation_id'] or '-'}",
        f"Message:     {decision['message_to_employee']}",
        rule,
    ]
    return "\n".join(lines)
