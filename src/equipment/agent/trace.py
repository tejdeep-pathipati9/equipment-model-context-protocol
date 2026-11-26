"""Thought / Action / Observation and reflection printer, compact enough to screenshot."""

import json

from equipment.agent.types import AgentResult, FinalDecision, Outcome, Review, Step
from equipment.domain.models import Decision


def format_step(number: int, step: Step) -> str:
    observation = json.dumps(step.observation)
    if step.is_error:
        observation = f"ERROR {observation}"
    actor = "  Actor:       code\n" if step.actor == "code" else ""
    return (
        f"Step {number}\n"
        f"{actor}"
        f"  Thought:     {step.thought or '(none)'}\n"
        f"  Action:      {step.action}({json.dumps(step.arguments)})\n"
        f"  Observation: {observation}"
    )


def format_review(number: int, review: Review) -> str:
    """One draft and its checks; printing every draft shows what each revision changed."""
    draft = review.draft
    if review.code_issues:
        critic = "not run"
    elif review.no_verdict:
        critic = "no verdict"
    else:
        critic = _result(review.issues)
    note = draft.get("escalation_reason")
    lines = [
        f"Reflection: draft {number}",
        f"  Decision:    {str(draft.get('decision', '-')).upper()} ({draft.get('reason_code')})",
        f"  Message:     {draft.get('message_to_employee', '-')}",
        *([f"  Reviewer:    {note}"] if note else []),
        f"  Code check:  {_result(review.code_issues)}",
        f"  Critic:      {critic}",
        *(f"    - {issue}" for issue in review.issues),
    ]
    return "\n".join(lines)


def format_outcome(result: AgentResult) -> str:
    """e.g. Reflection: REVISED (1 revision)"""
    line = f"Reflection: {result.outcome.upper()}"
    if result.outcome == Outcome.SKIPPED:
        return f"{line} (no draft was reviewed)"
    line += f" ({_count(len(result.reviews) - 1, 'revision')})"
    if result.outcome == Outcome.ESCALATED:
        line += f": {'; '.join(result.reviews[-1].issues)}"
    return line


def _result(issues: list[str]) -> str:
    return _count(len(issues), "issue") if issues else "pass"


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def format_decision(decision: FinalDecision) -> str:
    rule = "=" * 72
    lines = [
        rule,
        f"Decision:    {decision.decision.upper()} ({decision.reason_code})",
        f"Item:        {decision.item or '-'}",
        f"Escalation:  {_escalation(decision)}",
        f"Message:     {decision.message_to_employee}",
        rule,
    ]
    return "\n".join(lines)


def _escalation(decision: FinalDecision) -> str:
    if decision.escalation_id:
        return decision.escalation_id
    return "NOT CONFIRMED" if decision.decision == Decision.ESCALATE else "-"
