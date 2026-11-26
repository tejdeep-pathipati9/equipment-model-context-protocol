"""The code check: what code can prove from the tool results (requirements 5).

Runs before the critic, on every draft and on every escalation before it is saved.
"""

from typing import Any

from pydantic import ValidationError

from equipment.agent.types import Draft, Eligibility, Proposal, Step
from equipment.domain.models import Basis, Decision, ErrorCode, ReasonCode, Status

ELIGIBILITY = "check_request_eligibility"
FLAG = "flag_for_human_review"
VERDICTS = {Decision.APPROVE: Status.ELIGIBLE, Decision.DENY: Status.NOT_ELIGIBLE}
SET_BY_CODE = {ReasonCode.REFLECTION_FAILED, ReasonCode.AGENT_STEP_LIMIT, ReasonCode.AGENT_REFUSED}
NEEDS_REVIEW_CODES = {ReasonCode.ITEM_NOT_IN_CATALOG, ReasonCode.RECORDS_INCOMPLETE}


def check_draft(draft: Draft, steps: list[Step]) -> list[str]:
    """Problems code can prove, e.g. approve when the tool said not_eligible. Empty = pass."""
    if draft.decision == Decision.ESCALATE:
        return _check_escalation(draft, steps)
    return _check_verdict(draft, steps)


def check_escalation(proposal: Proposal, steps: list[Step]) -> list[str]:
    """Whether the reason code fits the tool results, in rule order (requirements 4).

    e.g. EXCEPTION_REQUEST when the employee may have one more unit -> approve instead.
    """
    code = proposal.reason_code
    if code in SET_BY_CODE:
        return [f"{code} is set by code, not by the agent"]
    if code == ReasonCode.UNKNOWN_EMPLOYEE:
        if any(_error(step) == ErrorCode.EMPLOYEE_NOT_FOUND for step in steps):
            return []
        return [f"{code}, but no lookup returned EMPLOYEE_NOT_FOUND"]
    if code == ReasonCode.VAGUE_REQUEST:  # rule 2 comes before every eligibility rule
        return []

    result = _last_eligibility(steps)
    if result is None:
        return [f"{code}, but there is no eligibility result"]
    status, verdict = result.status, result.reason_code
    if status == Status.NEEDS_REVIEW and code != verdict:
        return [f"{code}, but eligibility is needs_review ({verdict}): escalate with {verdict}"]
    if status != Status.NEEDS_REVIEW and code in NEEDS_REVIEW_CODES:
        return [f"{code}, but eligibility is {status} ({verdict})"]
    if code == ReasonCode.EXCEPTION_REQUEST and result.basis == Basis.QUANTITY:
        return [f"{code}, but policy allows one more unit ({verdict}): approve"]
    return []


def saved_escalation(steps: list[Step]) -> Step | None:
    """The step that saved an escalation in this run, e.g. ESC-3f9a1c2e, or None."""
    for step in steps:
        saved = isinstance(step.observation, dict) and step.observation.get("escalation_id")
        if step.action == FLAG and saved:
            return step
    return None


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
    if result.status == Status.NEEDS_REVIEW:  # requirements 4, rule 3 comes before rule 6
        return [f"{decision}, but eligibility is needs_review ({result.reason_code}): escalate"]

    issues = []
    if draft.escalation_id is not None:
        issues.append(f"{decision}, but escalation_id is {draft.escalation_id}")
    if draft.item != result.item:
        issues.append(f"item is {draft.item}, but eligibility was checked for {result.item}")
    if decision == Decision.DENY and code == ReasonCode.AT_MAX_QUANTITY:
        if result.basis != Basis.REFRESH:
            issues.append(f"{code} needs basis refresh, but the basis is {result.basis}")
    elif result.status != VERDICTS[decision]:
        issues.append(f"{decision}, but eligibility is {result.status}")
    elif code != result.reason_code:
        issues.append(f"reason code is {code}, but eligibility says {result.reason_code}")
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


def _error(step: Step) -> Any:
    return step.observation.get("error") if isinstance(step.observation, dict) else None


def _last_eligibility(steps: list[Step]) -> Eligibility | None:
    """The last complete eligibility result; errors and incomplete results do not count."""
    for step in reversed(steps):
        if step.action != ELIGIBILITY or step.is_error:
            continue
        try:
            return Eligibility.model_validate(step.observation)
        except ValidationError:
            continue
    return None
