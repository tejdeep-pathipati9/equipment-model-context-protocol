"""Types shared by the ReAct loop, the review and the trace."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from equipment.domain.models import Basis, Decision, ReasonCode, Status

# ---------------------------------------------------------------- model outputs


class ModelOutput(BaseModel):
    """What the model submits. Validated here because a strict tool can still be cut off."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Draft(ModelOutput):
    """The model's decision. It has no escalation ID: code saves an escalation after review."""

    decision: Decision
    reason_code: ReasonCode
    item: str | None
    extra_unit: bool = Field(
        description="true only if the employee asks to keep every unit they hold and add one more"
    )
    message_to_employee: str = Field(min_length=1)
    escalation_reason: str | None = Field(
        description="escalate only: a short note for the human reviewer; null otherwise"
    )

    @model_validator(mode="after")
    def _reason_only_for_an_escalation(self) -> "Draft":
        escalating = self.decision == Decision.ESCALATE
        if escalating and not self.escalation_reason:
            raise ValueError("an escalation needs an escalation_reason for the human reviewer")
        if not escalating and self.escalation_reason:
            raise ValueError(f"{self.decision} must not have an escalation_reason")
        return self


class CriticIssue(ModelOutput):
    quote: str = Field(description="exact words from the draft")
    problem: str


class CriticReview(ModelOutput):
    issues: list[CriticIssue]


# ---------------------------------------------------------------- tool results


class Eligibility(BaseModel):
    """The fields of a check_request_eligibility result that the code check relies on.

    The result has more fields (held, max_qty, ...); they are ignored here.
    """

    model_config = ConfigDict(frozen=True)

    item: str
    status: Status
    reason_code: ReasonCode
    basis: Basis | None


class Receipt(BaseModel):
    """What flag_for_human_review returns once it saved the escalation; else it is unconfirmed."""

    escalation_id: str = Field(min_length=1)
    status: Literal["open"]


def validation_issues(error: ValidationError) -> list[str]:
    """e.g. "decision: Input should be 'approve', 'deny' or 'escalate'"."""
    issues = []
    for e in error.errors():
        field = ".".join(map(str, e["loc"]))  # empty for a rule across fields
        issues.append(f"{field}: {e['msg']}" if field else e["msg"])
    return issues


# ---------------------------------------------------------------- the run


class FinalDecision(BaseModel):
    """What the employee gets: the accepted draft, and the ID of the escalation code saved."""

    model_config = ConfigDict(frozen=True)

    decision: Decision
    reason_code: ReasonCode
    item: str | None
    message_to_employee: str
    escalation_id: str | None = None

    @classmethod
    def from_draft(
        cls, draft: Draft, escalation_id: str | None = None, message: str | None = None
    ) -> "FinalDecision":
        return cls(
            decision=draft.decision,
            reason_code=draft.reason_code,
            item=draft.item,
            message_to_employee=message or draft.message_to_employee,
            escalation_id=escalation_id,
        )


@dataclass(frozen=True)
class Request:
    """Checked once, wherever it comes from: " E001 " -> "E001"; the text is kept as written."""

    employee_id: str  # from metadata, never from the text
    text: str

    def __post_init__(self) -> None:
        employee_id = (self.employee_id or "").strip()
        if not employee_id:
            raise ValueError("the request needs an employee ID")
        if not (self.text or "").strip():
            raise ValueError("the request needs text")
        object.__setattr__(self, "employee_id", employee_id)  # frozen, so set it directly


@dataclass(frozen=True)
class Step:
    thought: str
    action: str
    arguments: dict[str, Any]
    observation: Any
    is_error: bool
    actor: str = "agent"  # "code" for an action the model did not choose, e.g. saving
    same_turn_as: int | None = None  # the step whose model turn also requested this one


@dataclass(frozen=True)
class Review:
    """The checks of one draft. The critic only runs once the code check passes."""

    draft: dict[str, Any]  # as submitted; it may be invalid
    code_issues: list[str]
    critic_issues: list[str] | None = None  # None = not run, or no verdict

    @property
    def passed(self) -> bool:
        return not self.code_issues and self.critic_issues == []

    @property
    def no_verdict(self) -> bool:
        return not self.code_issues and self.critic_issues is None

    @property
    def issues(self) -> list[str]:
        if self.no_verdict:
            return ["the critic gave no verdict on the draft"]
        return self.code_issues or self.critic_issues or []


Critic = Callable[[Request, Draft, list[Step]], Awaitable[list[str] | None]]
"""The critic's issues with a draft: empty = pass, None = no verdict."""


class Stop(StrEnum):
    """How a run ended. Everything except SUBMITTED is escalated by code."""

    SUBMITTED = "submitted"
    STEP_LIMIT = "step_limit"
    REFUSED = "refused"
    REFLECTION_FAILED = "reflection_failed"


class Outcome(StrEnum):
    """What reflection did to the decision."""

    CONFIRMED = "confirmed"  # the first draft passed
    REVISED = "revised"  # a later draft passed
    ESCALATED = "escalated"  # no draft passed
    SKIPPED = "skipped"  # no draft was reviewed


@dataclass
class AgentResult:
    decision: FinalDecision
    stop: Stop
    steps: list[Step] = field(default_factory=list)
    reviews: list[Review] = field(default_factory=list)

    @property
    def recorded(self) -> bool:
        """False for an escalation whose save was not confirmed: no human may ever see it."""
        decision = self.decision
        return decision.decision != Decision.ESCALATE or decision.escalation_id is not None

    @property
    def outcome(self) -> Outcome:
        if not self.reviews:
            return Outcome.SKIPPED
        if self.stop != Stop.SUBMITTED:
            return Outcome.ESCALATED
        rejected = any(review.issues for review in self.reviews)
        return Outcome.REVISED if rejected else Outcome.CONFIRMED
