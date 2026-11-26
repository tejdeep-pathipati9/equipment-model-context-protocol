"""Reflection: code check, critic and trace. Scripted fake LLM, no Azure."""

import asyncio
import json

import pytest
from fakes import FakeLLM, call, reply, thought

from equipment.agent.checks import check_draft, check_escalation
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
from equipment.agent.types import AgentResult, Draft, Proposal, Request, Review, Step, Stop
from equipment.domain.models import Item


def draft(**changes):
    return Draft(
        **{
            "decision": "approve",
            "reason_code": "WITHIN_LIMIT",
            "item": "monitor",
            "message_to_employee": "Approved: one more monitor.",
            "escalation_id": None,
        }
        | changes
    )


def as_json(proposed):
    return proposed.model_dump(mode="json")


def escalation(**changes):
    return draft(
        **{
            "decision": "escalate",
            "reason_code": "EXCEPTION_REQUEST",
            "message_to_employee": "A human reviewer will look at this (ESC-1).",
            "escalation_id": "ESC-1",
        }
        | changes
    )


def eligibility(status="eligible", reason_code="WITHIN_LIMIT", basis="quantity", item="monitor"):
    observation = {"item": item, "status": status, "reason_code": reason_code, "basis": basis}
    return Step("", "check_request_eligibility", {"item": item}, observation, False)


def flagged(id="ESC-1", reason_code="EXCEPTION_REQUEST"):
    observation = {"escalation_id": id, "status": "open"}
    return Step("", "flag_for_human_review", {"reason_code": reason_code}, observation, False)


NOT_ELIGIBLE = eligibility("not_eligible", "ROLE_NOT_ELIGIBLE", basis=None)
MISSING_DATE_AT_MAX = eligibility("needs_review", "RECORDS_INCOMPLETE", "refresh")


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
            [eligibility("not_eligible", "REFRESH_NOT_DUE", "refresh")],
            id="deny extra unit at max, refresh not due",
        ),
        pytest.param(escalation(), [eligibility(), flagged()], id="escalate with saved record"),
        pytest.param(escalation(item=None), [flagged()], id="escalate without eligibility"),
        pytest.param(
            draft(),
            [NOT_ELIGIBLE, eligibility()],
            id="last eligibility result counts",
        ),
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
        pytest.param(draft(item="laptop"), [eligibility()], "monitor", id="wrong item"),
        pytest.param(
            draft(escalation_id="ESC-9"), [eligibility()], "ESC-9", id="invented escalation"
        ),
        pytest.param(draft(item=None), [eligibility()], "monitor", id="no item"),
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
        pytest.param(draft(), [eligibility(), flagged()], "ESC-1", id="approve after escalating"),
        pytest.param(
            draft(decision="deny", reason_code="ROLE_NOT_ELIGIBLE"),
            [NOT_ELIGIBLE, flagged()],
            "ESC-1",
            id="deny after escalating",
        ),
        pytest.param(escalation(), [eligibility()], "no escalation", id="nothing saved"),
        pytest.param(
            escalation(escalation_id="ESC-2"), [flagged()], "ESC-1", id="wrong escalation id"
        ),
        pytest.param(escalation(escalation_id=None), [flagged()], "ESC-1", id="no escalation id"),
        pytest.param(
            escalation(reason_code="VAGUE_REQUEST"),
            [flagged()],
            "EXCEPTION_REQUEST",
            id="code differs from the record",
        ),
        pytest.param(
            escalation(message_to_employee="Sent to a reviewer."),
            [flagged()],
            "message",
            id="id missing from the message",
        ),
    ],
)
def test_check_finds_the_problem(proposed, steps, problem):
    issues = check_draft(proposed, steps)

    assert len(issues) == 1
    assert problem in issues[0]


def test_failed_tool_results_do_not_count():
    unknown = Step("", "check_request_eligibility", {}, {"error": "EMPLOYEE_NOT_FOUND"}, False)
    text = Step("", "check_request_eligibility", {}, "status unknown", False)
    blocked = Step("", "check_request_eligibility", {}, {"status": "eligible"}, True)

    assert "no eligibility" in check_draft(draft(), [unknown, text, blocked])[0]


def test_check_reports_every_problem():
    proposed = draft(reason_code="REFRESH_DUE", item="laptop")

    assert len(check_draft(proposed, [eligibility()])) == 2


# ---------------------------------------------------------------- code check of an escalation

NOT_FOUND = Step("", "get_employee_info", {}, {"error": "EMPLOYEE_NOT_FOUND"}, False)
NOT_DUE = eligibility("not_eligible", "REFRESH_NOT_DUE", "refresh")


def proposal(reason_code):
    return Proposal(reason_code=reason_code, reason="Needs a human.")


@pytest.mark.parametrize(
    ("reason_code", "steps"),
    [
        pytest.param("UNKNOWN_EMPLOYEE", [NOT_FOUND], id="unknown employee"),
        pytest.param("VAGUE_REQUEST", [], id="vague, before any lookup"),
        pytest.param("VAGUE_REQUEST", [MISSING_DATE_AT_MAX], id="vague comes before needs_review"),
        pytest.param("RECORDS_INCOMPLETE", [MISSING_DATE_AT_MAX], id="needs_review code"),
        pytest.param("CLAIM_CONTRADICTS_RECORDS", [NOT_DUE], id="contradiction"),
        pytest.param("EXCEPTION_REQUEST", [NOT_DUE], id="exception, not due"),
        pytest.param("EXCEPTION_REQUEST", [NOT_ELIGIBLE], id="exception, role not eligible"),
    ],
)
def test_escalation_check_passes(reason_code, steps):
    assert check_escalation(proposal(reason_code), steps) == []


@pytest.mark.parametrize(
    ("reason_code", "steps", "problem"),
    [
        pytest.param("EXCEPTION_REQUEST", [eligibility()], "approve", id="policy allows it"),
        pytest.param("EXCEPTION_REQUEST", [], "no eligibility", id="exception without a check"),
        pytest.param(
            "UNKNOWN_EMPLOYEE", [eligibility()], "EMPLOYEE_NOT_FOUND", id="known employee"
        ),
        pytest.param(
            "CLAIM_CONTRADICTS_RECORDS",
            [MISSING_DATE_AT_MAX],
            "RECORDS_INCOMPLETE",
            id="needs_review comes first",
        ),
        pytest.param("ITEM_NOT_IN_CATALOG", [NOT_DUE], "not_eligible", id="item is in the catalog"),
        pytest.param("AGENT_STEP_LIMIT", [], "code", id="code-only reason"),
    ],
)
def test_escalation_check_finds_the_problem(reason_code, steps, problem):
    [issue] = check_escalation(proposal(reason_code), steps)

    assert problem in issue


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


def test_critic_judges_a_proposed_escalation_instead_of_a_draft():
    llm = FakeLLM(review())

    run_critique(llm, proposed=proposal("VAGUE_REQUEST"))

    [message] = llm.calls[0]["messages"]
    assert json.loads(message["content"])["proposed_escalation"] == {
        "reason_code": "VAGUE_REQUEST",
        "reason": "Needs a human.",
    }


def test_critic_prompt_explains_a_proposed_escalation():
    assert "proposed_escalation" in CRITIC


def test_format_review_shows_a_proposed_escalation():
    current = Review(
        {"decision": "escalate", "reason_code": "VAGUE_REQUEST", "reason": "Two items."}, []
    )

    assert format_review(2, current).splitlines()[:3] == [
        "Reflection: draft 2 (proposed escalation)",
        "  Decision:    ESCALATE (VAGUE_REQUEST)",
        "  Reason:      Two items.",
    ]


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
    unrecorded = draft(decision="escalate", reason_code="AGENT_STEP_LIMIT", item=None)

    async def agent(*args, **kwargs):
        return AgentResult(unrecorded, Stop.STEP_LIMIT)

    monkeypatch.setattr("equipment.agent.cli.run_agent", agent)
    return lambda: asyncio.run(run(REQUEST))


def test_cli_fails_when_the_escalation_was_not_recorded(unrecorded_run):
    assert unrecorded_run() == 1


def test_cli_says_the_escalation_was_not_recorded(unrecorded_run, capsys):
    unrecorded_run()

    assert "escalation not recorded" in capsys.readouterr().err
