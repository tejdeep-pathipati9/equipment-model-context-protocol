"""The code check: what code can prove about a draft from the tool results (requirements 5).

Runs on every draft, before the critic. Each decision is judged by the eligibility result for the
draft's own item, so a lookup for another item never counts.
"""

from typing import Any

from pydantic import ValidationError

from equipment.agent.types import Draft, Eligibility, Step
from equipment.domain.models import Basis, Decision, ErrorCode, ReasonCode, Status

ELIGIBILITY = "check_request_eligibility"
VERDICTS = {Decision.APPROVE: Status.ELIGIBLE, Decision.DENY: Status.NOT_ELIGIBLE}
SET_BY_CODE = {ReasonCode.REFLECTION_FAILED, ReasonCode.AGENT_STEP_LIMIT, ReasonCode.AGENT_REFUSED}
NEEDS_REVIEW_CODES = {ReasonCode.ITEM_NOT_IN_CATALOG, ReasonCode.RECORDS_INCOMPLETE}


def check_draft(draft: Draft, steps: list[Step]) -> list[str]:
    """Problems code can prove, e.g. approve when the tool said not_eligible. Empty = pass."""
    if draft.decision == Decision.ESCALATE:
        return _check_escalation(draft, steps)
    return _check_verdict(draft, steps)


def eligibility_for(item: str | None, steps: list[Step]) -> Eligibility | None:
    """The last complete eligibility result for this item; errors and other items do not count."""
    wanted = (item or "").strip().lower()
    for step in reversed(steps):
        if step.action != ELIGIBILITY or step.is_error:
            continue
        try:
            result = Eligibility.model_validate(step.observation)
        except ValidationError:
            continue
        if result.item == wanted:
            return result
    return None


def _check_verdict(draft: Draft, steps: list[Step]) -> list[str]:
    """Approve or deny must match the eligibility result for the draft's item."""
    decision, code = draft.decision, draft.reason_code
    result = eligibility_for(draft.item, steps)
    if result is None:
        return [f"{decision}, but there is no eligibility result for {draft.item or 'the item'}"]
    if result.status == Status.NEEDS_REVIEW:  # requirements 4, rule 3 comes before rule 6
        return [f"{decision}, but eligibility is needs_review ({result.reason_code}): escalate"]
    if decision == Decision.DENY and code == ReasonCode.AT_MAX_QUANTITY:
        if result.basis != Basis.REFRESH:
            return [f"{code} needs basis refresh, but the basis is {result.basis}"]
        return []
    if result.status != VERDICTS[decision]:
        return [f"{decision}, but eligibility is {result.status}"]
    if code != result.reason_code:
        return [f"reason code is {code}, but eligibility says {result.reason_code}"]
    return []


def _check_escalation(draft: Draft, steps: list[Step]) -> list[str]:
    """Whether the reason code fits the tool results, in rule order (requirements 4).

    e.g. EXCEPTION_REQUEST when the employee may have one more unit -> approve instead.
    """
    code = draft.reason_code
    if code in SET_BY_CODE:
        return [f"{code} is set by code, not by the agent"]
    not_found = any(_error(step) == ErrorCode.EMPLOYEE_NOT_FOUND for step in steps)
    if code == ReasonCode.UNKNOWN_EMPLOYEE:
        return [] if not_found else [f"{code}, but no lookup returned EMPLOYEE_NOT_FOUND"]
    if not_found:  # rule 1 comes before every other rule
        return [f"{code}, but the employee was not found: escalate with UNKNOWN_EMPLOYEE"]
    if code == ReasonCode.VAGUE_REQUEST:  # rule 2 comes before every eligibility rule
        return []

    result = eligibility_for(draft.item, steps)
    if result is None:
        return [f"{code}, but there is no eligibility result for {draft.item or 'the item'}"]
    status, verdict = result.status, result.reason_code
    if status == Status.NEEDS_REVIEW and code != verdict:
        return [f"{code}, but eligibility is needs_review ({verdict}): escalate with {verdict}"]
    if status != Status.NEEDS_REVIEW and code in NEEDS_REVIEW_CODES:
        return [f"{code}, but eligibility is {status} ({verdict})"]
    if code == ReasonCode.EXCEPTION_REQUEST and result.basis == Basis.QUANTITY:
        return [f"{code}, but policy allows one more unit ({verdict}): approve"]
    return []


def _error(step: Step) -> Any:
    return step.observation.get("error") if isinstance(step.observation, dict) else None
