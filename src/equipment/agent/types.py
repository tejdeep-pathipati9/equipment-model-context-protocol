"""Types shared by the ReAct loop, the review and the trace."""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from equipment.domain.models import Decision, ReasonCode

# ---------------------------------------------------------------- model outputs


class ModelOutput(BaseModel):
    """What the model submits. Validated here because a strict tool can still be cut off."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class Draft(ModelOutput):
    decision: Decision
    reason_code: ReasonCode
    item: str | None
    message_to_employee: str = Field(min_length=1)
    escalation_id: str | None


class Proposal(ModelOutput):
    """An escalation the model asked to save; reviewed before flag_for_human_review runs."""

    reason_code: ReasonCode
    reason: str = Field(min_length=1)


class CriticIssue(ModelOutput):
    quote: str = Field(description="exact words from the draft")
    problem: str


class CriticReview(ModelOutput):
    issues: list[CriticIssue]


def validation_issues(error: ValidationError) -> list[str]:
    """e.g. "decision: Input should be 'approve', 'deny' or 'escalate'"."""
    return [f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in error.errors()]


# ---------------------------------------------------------------- the run


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


Critic = Callable[[Request, Draft | Proposal, list[Step]], Awaitable[list[str] | None]]
"""The critic's issues with a draft or a proposed escalation: empty = pass, None = no verdict."""


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
    decision: Draft
    stop: Stop
    steps: list[Step] = field(default_factory=list)
    reviews: list[Review] = field(default_factory=list)

    @property
    def recorded(self) -> bool:
        """False for an escalation that could not be saved: no human will ever see it."""
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
