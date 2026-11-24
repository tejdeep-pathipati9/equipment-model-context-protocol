"""Domain dataclasses and enums (docs/requirements.md 1, 2, 4, 6)."""

from dataclasses import dataclass
from datetime import date, datetime
from enum import StrEnum

# ---------------------------------------------------------------- vocabulary


class Role(StrEnum):
    INTERN = "intern"
    EMPLOYEE = "employee"
    MANAGER = "manager"
    CONTRACTOR = "contractor"


class Item(StrEnum):
    LAPTOP = "laptop"
    MONITOR = "monitor"
    KEYBOARD_MOUSE = "keyboard_mouse"
    HEADSET = "headset"
    PHONE = "phone"


# ---------------------------------------------------------------- records


@dataclass(frozen=True)
class Employee:
    id: str
    name: str
    role: str  # stored text; an unknown role must surface as data, not crash on load
    hire_date: date


@dataclass(frozen=True)
class EquipmentUnit:
    item: str
    issued_on: date | None  # None = missing issue date (escalated as RECORDS_INCOMPLETE)


@dataclass(frozen=True)
class PolicyLimit:
    item: str
    max_qty: int  # 0 = role not eligible for the item
    refresh_months: int


# ---------------------------------------------------------------- eligibility (requirements §2.1)


class Status(StrEnum):
    ELIGIBLE = "eligible"
    NOT_ELIGIBLE = "not_eligible"
    NEEDS_REVIEW = "needs_review"


class Basis(StrEnum):
    """Which check ran, inferred from holdings: below max -> quantity, at max -> refresh."""

    QUANTITY = "quantity"
    REFRESH = "refresh"


class ReasonCode(StrEnum):
    """Every reason code a final response can carry (one enum, requirements §2.1, §4, §5)."""

    # eligibility verdicts
    WITHIN_LIMIT = "WITHIN_LIMIT"
    REFRESH_DUE = "REFRESH_DUE"
    REFRESH_NOT_DUE = "REFRESH_NOT_DUE"
    ROLE_NOT_ELIGIBLE = "ROLE_NOT_ELIGIBLE"
    ITEM_NOT_IN_CATALOG = "ITEM_NOT_IN_CATALOG"
    RECORDS_INCOMPLETE = "RECORDS_INCOMPLETE"
    # agent decisions
    UNKNOWN_EMPLOYEE = "UNKNOWN_EMPLOYEE"
    VAGUE_REQUEST = "VAGUE_REQUEST"
    CLAIM_CONTRADICTS_RECORDS = "CLAIM_CONTRADICTS_RECORDS"
    EXCEPTION_REQUEST = "EXCEPTION_REQUEST"
    AT_MAX_QUANTITY = "AT_MAX_QUANTITY"
    # safe fallbacks
    REFLECTION_FAILED = "REFLECTION_FAILED"
    AGENT_STEP_LIMIT = "AGENT_STEP_LIMIT"


# The reason codes that may be escalated (the rest are approve/deny outcomes).
ESCALATION_CODES = frozenset(
    {
        ReasonCode.UNKNOWN_EMPLOYEE,
        ReasonCode.VAGUE_REQUEST,
        ReasonCode.ITEM_NOT_IN_CATALOG,
        ReasonCode.RECORDS_INCOMPLETE,
        ReasonCode.CLAIM_CONTRADICTS_RECORDS,
        ReasonCode.EXCEPTION_REQUEST,
        ReasonCode.REFLECTION_FAILED,
        ReasonCode.AGENT_STEP_LIMIT,
    }
)


@dataclass(frozen=True)
class Escalation:
    id: str
    employee_id: str  # may be unknown: unknown employees are escalated too
    request: str
    reason: str
    reason_code: ReasonCode
    created_at: datetime
    status: str = "open"


@dataclass(frozen=True)
class EligibilityResult:
    status: Status
    reason_code: ReasonCode
    basis: Basis | None = None  # None when no quantity/refresh check ran (rules 1-3)
    held: int | None = None
    max_qty: int | None = None
    oldest_age_months: int | None = None
    refresh_months: int | None = None
    missing: str | None = None  # "role_policy" | "issue_date" for RECORDS_INCOMPLETE


# ---------------------------------------------------------------- tool errors and decisions


class ErrorCode(StrEnum):
    """Structured errors returned by tools as data, never raised across MCP."""

    EMPLOYEE_NOT_FOUND = "EMPLOYEE_NOT_FOUND"
    INVALID_EMPLOYEE_ID = "INVALID_EMPLOYEE_ID"
    UNKNOWN_ROLE = "UNKNOWN_ROLE"
    INVALID_ESCALATION = "INVALID_ESCALATION"


class Decision(StrEnum):
    APPROVE = "approve"
    DENY = "deny"
    ESCALATE = "escalate"
