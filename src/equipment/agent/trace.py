"""Thought / Action / Observation and reflection printer, compact enough to screenshot."""

import json
from typing import Any

from equipment.agent.react import Step
from equipment.agent.reflection import Outcome, Reflection, Round


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


def format_round(number: int, current: Round) -> str:
    """One draft and its checks; printing every draft shows what each revision changed."""
    draft = current.draft
    if current.code_issues:
        critic = "not run"
    elif current.no_verdict:
        critic = "no verdict"
    else:
        critic = _result(current.issues)
    lines = [
        f"Reflection: draft {number}",
        f"  Decision:    {str(draft['decision']).upper()} ({draft['reason_code']})",
        f"  Message:     {draft['message_to_employee']}",
        f"  Code check:  {_result(current.code_issues)}",
        f"  Critic:      {critic}",
        *(f"    - {issue}" for issue in current.issues),
    ]
    return "\n".join(lines)


def format_outcome(reflection: Reflection) -> str:
    """e.g. Reflection: REVISED (1 revision)"""
    line = f"Reflection: {reflection.outcome.upper()}"
    if reflection.outcome == Outcome.SKIPPED:
        return f"{line} (decision made by code)"
    line += f" ({_count(len(reflection.rounds) - 1, 'revision')})"
    return f"{line}: {reflection.reason}" if reflection.reason else line


def _result(issues: list[str]) -> str:
    return _count(len(issues), "issue") if issues else "pass"


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


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
