"""Reflection: code check, critic and trace. Scripted fake LLM, no Azure."""

import asyncio
import json

import pytest
from fakes import FakeLLM, call, reply, thought
from pydantic import ValidationError

from equipment.agent.checks import check_draft
from equipment.agent.cli import run
from equipment.agent.prompts import (
    CATALOG,
    CRITIC,
    DECISION_RULES,
    MESSAGE_RULES,
    REJECTED,
    SYSTEM,
    SYSTEM_FACTS,
)
from equipment.agent.reflection import REVIEW_TOOL, critique
from equipment.agent.trace import format_outcome, format_review
from equipment.agent.types import AgentResult, Draft, FinalDecision, Request, Review, Step, Stop
from equipment.domain.models import Item


def draft(**changes):
    return Draft(
        **{
            "decision": "approve",
            "reason_code": "WITHIN_LIMIT",
            "item": "monitor",
            "message_to_employee": "Approved: one more monitor.",
            "escalation_reason": None,
        }
        | changes
    )


def as_json(proposed):
    return proposed.model_dump(mode="json")


def escalation(reason_code="EXCEPTION_REQUEST", item="monitor"):
    return draft(
        decision="escalate",
        reason_code=reason_code,
        item=item,
        message_to_employee="Your monitor is not due for a refresh, so a person needs to decide.",
        escalation_reason="Employee says the monitor broke.",
    )


def eligibility(status="eligible", reason_code="WITHIN_LIMIT", basis="quantity", item="monitor"):
    observation = {"item": item, "status": status, "reason_code": reason_code, "basis": basis}
    return Step("", "check_request_eligibility", {"item": item}, observation, False)


NOT_ELIGIBLE = eligibility("not_eligible", "ROLE_NOT_ELIGIBLE", basis=None)
MISSING_DATE_AT_MAX = eligibility("needs_review", "RECORDS_INCOMPLETE", "refresh")
NOT_DUE = eligibility("not_eligible", "REFRESH_NOT_DUE", "refresh")
NOT_FOUND = Step("", "get_employee_info", {}, {"error": "EMPLOYEE_NOT_FOUND"}, False)
LOOKED_UP = Step("", "get_employee_info", {}, {"id": "E001", "role": "employee"}, False)
INVALID_ID = Step("", "get_employee_info", {}, {"error": "INVALID_EMPLOYEE_ID"}, False)
HEADSET = eligibility("not_eligible", "REFRESH_NOT_DUE", "refresh", item="headset")


# ---------------------------------------------------------------- the draft


@pytest.mark.parametrize(
    ("changes", "problem"),
    [
        pytest.param(
            {"decision": "escalate"}, "needs an escalation_reason", id="escalate, no note"
        ),
        pytest.param({"escalation_reason": "x"}, "must not have", id="approve with a note"),
    ],
)
def test_escalation_reason_only_comes_with_an_escalation(changes, problem):
    with pytest.raises(ValidationError, match=problem):
        draft(**changes)


def test_draft_has_no_escalation_id():
    assert "escalation_id" not in Draft.model_fields


# ---------------------------------------------------------------- code check


@pytest.mark.parametrize(
    ("proposed", "steps"),
    [
        pytest.param(draft(), [eligibility()], id="approve eligible"),
        pytest.param(
            draft(decision="deny", reason_code="ROLE_NOT_ELIGIBLE"),
            [NOT_ELIGIBLE],
            id="deny not eligible",
        ),
        pytest.param(
            draft(decision="deny", reason_code="AT_MAX_QUANTITY"),
            [eligibility("eligible", "REFRESH_DUE", "refresh")],
            id="deny extra unit at max, refresh due",
        ),
        pytest.param(
            draft(decision="deny", reason_code="AT_MAX_QUANTITY"),
            [NOT_DUE],
            id="deny extra unit at max, refresh not due",
        ),
        pytest.param(draft(), [NOT_ELIGIBLE, eligibility()], id="last result for the item counts"),
        pytest.param(draft(), [eligibility(), HEADSET], id="a later lookup for another item"),
        pytest.param(escalation("UNKNOWN_EMPLOYEE", None), [NOT_FOUND], id="unknown employee"),
        pytest.param(escalation("VAGUE_REQUEST", None), [LOOKED_UP], id="vague, after a lookup"),
        pytest.param(escalation("VAGUE_REQUEST"), [MISSING_DATE_AT_MAX], id="vague before rule 3"),
        pytest.param(escalation("RECORDS_INCOMPLETE"), [MISSING_DATE_AT_MAX], id="needs_review"),
        pytest.param(escalation("CLAIM_CONTRADICTS_RECORDS"), [NOT_DUE], id="contradiction"),
        pytest.param(escalation(), [NOT_DUE], id="exception, not due"),
        pytest.param(escalation(), [NOT_ELIGIBLE], id="exception, role not eligible"),
        pytest.param(escalation(), [NOT_DUE, HEADSET], id="exception, then another item"),
    ],
)
def test_check_passes(proposed, steps):
    assert check_draft(proposed, steps) == []


@pytest.mark.parametrize(
    ("proposed", "steps", "problem"),
    [
        pytest.param(draft(), [], "no eligibility", id="approve without a check"),
        pytest.param(draft(), [NOT_ELIGIBLE], "not_eligible", id="approve not eligible"),
        pytest.param(
            draft(reason_code="REFRESH_DUE"), [eligibility()], "WITHIN_LIMIT", id="wrong code"
        ),
        pytest.param(draft(item="laptop"), [eligibility()], "for laptop", id="other item checked"),
        pytest.param(draft(), [HEADSET], "for monitor", id="only another item checked"),
        pytest.param(draft(item=None), [eligibility()], "for the item", id="no item"),
        pytest.param(
            draft(decision="deny"), [eligibility()], "eligible", id="deny an eligible request"
        ),
        pytest.param(
            draft(decision="deny", reason_code="AT_MAX_QUANTITY"),
            [eligibility()],
            "refresh",
            id="at max on basis quantity",
        ),
        pytest.param(
            draft(reason_code="AT_MAX_QUANTITY"),
            [eligibility("eligible", "REFRESH_DUE", "refresh")],
            "REFRESH_DUE",
            id="approve with a deny code",
        ),
        pytest.param(
            draft(reason_code="RECORDS_INCOMPLETE"),
            [eligibility("needs_review", "RECORDS_INCOMPLETE", None)],
            "needs_review",
            id="approve needs_review",
        ),
        pytest.param(
            draft(decision="deny", reason_code="RECORDS_INCOMPLETE"),
            [eligibility("needs_review", "RECORDS_INCOMPLETE", None)],
            "needs_review",
            id="deny needs_review",
        ),
        pytest.param(
            draft(decision="deny", reason_code="AT_MAX_QUANTITY"),
            [MISSING_DATE_AT_MAX],
            "needs_review",
            id="at max, but needs_review",
        ),
        pytest.param(
            draft(reason_code="RECORDS_INCOMPLETE"),
            [MISSING_DATE_AT_MAX],
            "needs_review",
            id="approve needs_review on basis refresh",
        ),
        pytest.param(escalation(), [eligibility()], "approve", id="exception, policy allows it"),
        pytest.param(escalation(), [], "no eligibility", id="exception without a check"),
        pytest.param(escalation(), [HEADSET], "for monitor", id="exception, other item checked"),
        pytest.param(
            escalation("UNKNOWN_EMPLOYEE"), [eligibility()], "EMPLOYEE_NOT_FOUND", id="known"
        ),
        pytest.param(
            escalation("CLAIM_CONTRADICTS_RECORDS"),
            [MISSING_DATE_AT_MAX],
            "RECORDS_INCOMPLETE",
            id="needs_review comes first",
        ),
        pytest.param(
            escalation("ITEM_NOT_IN_CATALOG"), [NOT_DUE], "not_eligible", id="item in catalog"
        ),
        pytest.param(escalation("AGENT_STEP_LIMIT"), [], "code", id="code-only reason"),
        pytest.param(escalation("REFRESH_NOT_DUE"), [NOT_DUE], "not an escalation", id="deny code"),
        pytest.param(
            escalation("VAGUE_REQUEST", None), [], "looked up", id="vague, before rule 1 ran"
        ),
        pytest.param(
            escalation("VAGUE_REQUEST", None), [INVALID_ID], "looked up", id="vague, lookup failed"
        ),
        pytest.param(
            escalation("VAGUE_REQUEST"),
            [NOT_FOUND],
            "UNKNOWN_EMPLOYEE",
            id="unknown employee comes first",
        ),
    ],
)
def test_check_finds_the_problem(proposed, steps, problem):
    [issue] = check_draft(proposed, steps)

    assert problem in issue


def test_failed_tool_results_do_not_count():
    unknown = Step("", "check_request_eligibility", {}, {"error": "EMPLOYEE_NOT_FOUND"}, False)
    text = Step("", "check_request_eligibility", {}, "status unknown", False)
    blocked = Step("", "check_request_eligibility", {}, {"status": "eligible"}, True)

    assert "no eligibility" in check_draft(draft(), [unknown, text, blocked])[0]


@pytest.mark.parametrize(
    "observation",
    [
        pytest.param({"status": "eligible", "reason_code": "WITHIN_LIMIT"}, id="no item"),
        pytest.param({"item": "monitor", "status": "eligible"}, id="no reason code"),
        pytest.param(
            {"item": "monitor", "status": "maybe", "reason_code": "WITHIN_LIMIT"},
            id="unknown status",
        ),
    ],
)
def test_incomplete_eligibility_result_does_not_count(observation):
    incomplete = Step("", "check_request_eligibility", {}, observation, False)

    assert check_draft(draft(), [incomplete]) == [
        "approve, but there is no eligibility result for monitor"
    ]


# ---------------------------------------------------------------- critic

REQUEST = Request("E001", "Can I get a second monitor by Friday?")
BY_FRIDAY = {"quote": "by Friday", "problem": "No tool result mentions a date."}


def review(*issues):
    return reply(call("submit_review", {"issues": list(issues)}))


def submit(proposed):
    return reply(call("submit_decision", as_json(proposed)))


def run_critique(llm, proposed=None, steps=None):
    return asyncio.run(critique(llm, REQUEST, proposed or draft(), steps or [eligibility()]))


def test_critic_passes_a_draft_without_issues():
    assert run_critique(FakeLLM(review())) == []


def test_critic_issue_shows_the_quote_and_the_problem():
    assert run_critique(FakeLLM(review(BY_FRIDAY))) == [
        '"by Friday": No tool result mentions a date.'
    ]


def test_critic_sees_request_draft_and_results_but_not_thoughts():
    step = Step("SECRET REASONING", "get_employee_info", {"employee_id": "E001"}, {"n": 1}, False)
    submitted = Step("SECRET REASONING", "submit_decision", {}, {"accepted": True}, False)
    llm = FakeLLM(review())

    run_critique(llm, steps=[step, submitted])

    [sent] = llm.calls
    [message] = sent["messages"]
    assert json.loads(message["content"]) == {
        "request": {"employee_id": "E001", "text": REQUEST.text},
        "tool_results": [
            {"tool": "get_employee_info", "arguments": {"employee_id": "E001"}, "result": {"n": 1}}
        ],
        "draft": as_json(draft()),
    }
    assert sent["system"] == CRITIC
    assert sent["tools"] == [REVIEW_TOOL]


@pytest.mark.parametrize(
    "no_verdict",
    [
        pytest.param(reply(stop="refusal"), id="refusal"),
        pytest.param(reply(thought("Looks fine."), stop="end_turn"), id="text only"),
        pytest.param(reply(call("submit_review", {}), stop="max_tokens"), id="cut off"),
        pytest.param(reply(call("other_tool", {})), id="wrong tool"),
        pytest.param(reply(call("submit_review", {})), id="no issues field"),
        pytest.param(
            reply(call("submit_review", {"issues": ["bad"]})), id="issue is not an object"
        ),
        pytest.param(
            reply(call("submit_review", {"issues": [{"quote": "x"}]})), id="issue without a problem"
        ),
    ],
)
def test_critic_without_a_verdict_is_not_a_pass(no_verdict):
    assert run_critique(FakeLLM(no_verdict)) is None


def test_review_tool_is_strict():
    schema = REVIEW_TOOL["input_schema"]

    assert REVIEW_TOOL["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["issues"]


def test_drafter_and_critic_share_the_decision_rules():
    assert all(DECISION_RULES in prompt for prompt in (SYSTEM, CRITIC))


@pytest.mark.parametrize("prompt", [SYSTEM, CRITIC, REJECTED])
def test_prompts_tell_a_current_state_from_a_promised_action(prompt):
    assert "a reviewer will look at it" in prompt


def test_critic_and_rejection_know_the_system_facts():
    assert all(SYSTEM_FACTS in prompt for prompt in (CRITIC, REJECTED))
    assert all(MESSAGE_RULES in prompt for prompt in (CRITIC, REJECTED))
    assert all(CATALOG in prompt for prompt in (SYSTEM, SYSTEM_FACTS))


def test_catalog_matches_the_domain():
    assert CATALOG.split(", ") == list(Item)


# ---------------------------------------------------------------- the review


def test_no_verdict_is_an_issue():
    current = Review(as_json(draft()), [], None)

    assert (current.passed, current.no_verdict) == (False, True)
    assert "no verdict" in current.issues[0]


# ---------------------------------------------------------------- trace

BAD = draft(message_to_employee="Approved. You will have it by Friday.")
GOOD = draft(message_to_employee="Approved. I cannot confirm a delivery date.")


def test_format_review_shows_the_draft_and_the_critic_issues():
    current = Review(as_json(BAD), [], ['"by Friday": No tool result mentions a date.'])

    assert format_review(1, current) == (
        "Reflection: draft 1\n"
        "  Decision:    APPROVE (WITHIN_LIMIT)\n"
        "  Message:     Approved. You will have it by Friday.\n"
        "  Code check:  pass\n"
        "  Critic:      1 issue\n"
        '    - "by Friday": No tool result mentions a date.'
    )


def test_critic_checks_the_note_for_the_reviewer():
    assert "escalation_reason" in CRITIC


def test_format_review_shows_the_note_for_the_reviewer():
    current = Review(as_json(escalation("VAGUE_REQUEST", None)), [])

    assert "  Reviewer:    Employee says the monitor broke." in format_review(1, current)


@pytest.mark.parametrize(
    ("current", "code_line", "critic_line"),
    [
        (Review(as_json(GOOD), [], []), "Code check:  pass", "Critic:      pass"),
        (Review(as_json(GOOD), ["a", "b"]), "Code check:  2 issues", "Critic:      not run"),
        (Review(as_json(GOOD), [], None), "Code check:  pass", "Critic:      no verdict"),
    ],
)
def test_format_review_shows_each_check(current, code_line, critic_line):
    lines = [line.strip() for line in format_review(1, current).splitlines()]

    assert code_line in lines
    assert critic_line in lines


FAILED = Review(as_json(BAD), [], ["a", "b"])
PASSED = Review(as_json(GOOD), [], [])


@pytest.mark.parametrize(
    ("stop", "reviews", "expected"),
    [
        (Stop.SUBMITTED, [PASSED], "Reflection: CONFIRMED (0 revisions)"),
        (Stop.SUBMITTED, [FAILED, PASSED], "Reflection: REVISED (1 revision)"),
        (Stop.REFLECTION_FAILED, [FAILED] * 3, "Reflection: ESCALATED (2 revisions): a; b"),
        (Stop.STEP_LIMIT, [FAILED], "Reflection: ESCALATED (0 revisions): a; b"),
        (Stop.REFUSED, [], "Reflection: SKIPPED (no draft was reviewed)"),
    ],
)
def test_format_outcome(stop, reviews, expected):
    assert format_outcome(AgentResult(GOOD, stop, [], reviews)) == expected


def test_review_of_an_incomplete_draft_can_be_printed():
    assert "Message:     -" in format_review(1, Review({"decision": "approve"}, ["missing: item"]))


def test_cli_prints_the_reflected_decision(server_db, monkeypatch, capsys):
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    eligible = call("check_request_eligibility", {"employee_id": "E001", "item": "monitor"})
    llm = FakeLLM(reply(eligible), submit(BAD), review(BY_FRIDAY), submit(GOOD), review())
    monkeypatch.setattr("equipment.agent.cli.FoundryClient.from_env", lambda: llm)

    code = asyncio.run(run(REQUEST))

    output = capsys.readouterr().out
    final = output[output.index("Reflection: REVISED") :]
    assert code == 0
    assert "Reflection: draft 1" in output and "Reflection: draft 2" in output
    assert GOOD.message_to_employee in final
    assert BAD.message_to_employee not in final


@pytest.fixture
def unrecorded_run(server_db, monkeypatch):
    """The CLI, with an agent whose escalation could not be saved."""
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    monkeypatch.setattr("equipment.agent.cli.FoundryClient.from_env", FakeLLM)
    unrecorded = FinalDecision(
        decision="escalate",
        reason_code="AGENT_STEP_LIMIT",
        item=None,
        message_to_employee="Your request could not be recorded for human review.",
    )

    async def agent(*args, **kwargs):
        return AgentResult(unrecorded, Stop.STEP_LIMIT)

    monkeypatch.setattr("equipment.agent.cli.run_agent", agent)
    return lambda: asyncio.run(run(REQUEST))


def test_cli_fails_when_the_escalation_was_not_recorded(unrecorded_run):
    assert unrecorded_run() == 1


def test_cli_says_the_escalation_was_not_recorded(unrecorded_run, capsys):
    unrecorded_run()

    assert "escalation not confirmed" in capsys.readouterr().err
