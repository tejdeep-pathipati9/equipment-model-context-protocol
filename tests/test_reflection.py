"""Reflection: code check, critic and revisions. Scripted fake LLM, no Azure."""

import asyncio
import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest
from fakes import FakeLLM, call, reply, thought

from equipment.agent.cli import run
from equipment.agent.mcp_client import connect
from equipment.agent.prompts import CATALOG, CRITIC, DECISION_RULES, REVISE, SYSTEM, SYSTEM_FACTS
from equipment.agent.react import SUBMIT_TOOL, AgentResult, Request, Step
from equipment.agent.reflection import (
    REVIEW_TOOL,
    Outcome,
    Reflection,
    Round,
    check_draft,
    critique,
    reflect,
)
from equipment.agent.trace import format_outcome, format_round
from equipment.domain.models import Item


def draft(**changes):
    return {
        "decision": "approve",
        "reason_code": "WITHIN_LIMIT",
        "item": "monitor",
        "message_to_employee": "Approved: one more monitor.",
        "escalation_id": None,
    } | changes


def escalation(**changes):
    return (
        draft(
            decision="escalate",
            reason_code="EXCEPTION_REQUEST",
            message_to_employee="A human reviewer will look at this (ESC-1).",
            escalation_id="ESC-1",
        )
        | changes
    )


def eligibility(status="eligible", reason_code="WITHIN_LIMIT", basis="quantity", item="monitor"):
    observation = {"item": item, "status": status, "reason_code": reason_code, "basis": basis}
    return Step("", "check_request_eligibility", {"item": item}, observation, False)


def flagged(id="ESC-1", reason_code="EXCEPTION_REQUEST"):
    observation = {"escalation_id": id, "status": "open"}
    return Step("", "flag_for_human_review", {"reason_code": reason_code}, observation, False)


NOT_ELIGIBLE = eligibility("not_eligible", "ROLE_NOT_ELIGIBLE", basis=None)


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


# ---------------------------------------------------------------- critic

REQUEST = Request("E001", "Can I get a second monitor by Friday?")
BY_FRIDAY = {"quote": "by Friday", "problem": "No tool result mentions a date."}


def review(*issues):
    return reply(call("submit_review", {"issues": list(issues)}))


def revise(proposed):
    return reply(call("submit_decision", proposed))


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
    submitted = Step("SECRET REASONING", "submit_decision", draft(), {"accepted": True}, False)
    llm = FakeLLM(review())

    run_critique(llm, steps=[step, submitted])

    [sent] = llm.calls
    [message] = sent["messages"]
    assert json.loads(message["content"]) == {
        "request": {"employee_id": "E001", "text": REQUEST.text},
        "tool_results": [
            {"tool": "get_employee_info", "arguments": {"employee_id": "E001"}, "result": {"n": 1}}
        ],
        "draft": draft(),
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
    ],
)
def test_critic_without_a_verdict_is_not_a_pass(no_verdict):
    assert run_critique(FakeLLM(no_verdict)) is None


def test_review_tool_is_strict():
    schema = REVIEW_TOOL["input_schema"]

    assert REVIEW_TOOL["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["issues"]


def test_drafter_critic_and_reviser_share_the_decision_rules():
    assert all(DECISION_RULES in prompt for prompt in (SYSTEM, CRITIC, REVISE))


def test_critic_and_reviser_know_the_system_facts():
    assert all(SYSTEM_FACTS in prompt for prompt in (CRITIC, REVISE))
    assert all(CATALOG in prompt for prompt in (SYSTEM, SYSTEM_FACTS))


def test_catalog_matches_the_domain():
    assert CATALOG.split(", ") == list(Item)


# ---------------------------------------------------------------- the reflection loop


class FakeMCP:
    """Saves every escalation as ESC-NEW."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        saved = {"escalation_id": "ESC-NEW", "status": "open"}
        return SimpleNamespace(structured_content=saved, content=[], is_error=False)


def run_reflect(llm, proposed=None, steps=None, stop="submitted", mcp=None, **options):
    result = AgentResult(proposed or draft(), stop, steps or [eligibility()])
    return asyncio.run(reflect(llm, mcp or FakeMCP(), REQUEST, result, **options))


BAD = draft(message_to_employee="Approved. You will have it by Friday.")
GOOD = draft(message_to_employee="Approved. I cannot confirm a delivery date.")


def test_clean_draft_is_confirmed():
    llm = FakeLLM(review())

    reflection = run_reflect(llm, GOOD)

    assert (reflection.outcome, reflection.decision) == (Outcome.CONFIRMED, GOOD)
    assert len(llm.calls) == 1


def test_critic_issue_is_revised_and_checked_again():
    llm = FakeLLM(review(BY_FRIDAY), revise(GOOD), review())

    reflection = run_reflect(llm, BAD)

    assert (reflection.outcome, reflection.decision) == (Outcome.REVISED, GOOD)
    first, second = reflection.rounds
    assert (first.draft, len(first.critic_issues)) == (BAD, 1)
    assert (second.draft, second.critic_issues) == (GOOD, [])


def test_failed_code_check_skips_the_critic():
    denied = draft(decision="deny", reason_code="ROLE_NOT_ELIGIBLE", item="laptop")
    not_eligible = eligibility("not_eligible", "ROLE_NOT_ELIGIBLE", None, item="laptop")
    llm = FakeLLM(revise(denied), review())

    reflection = run_reflect(llm, draft(item="laptop"), [not_eligible])

    assert (reflection.outcome, reflection.decision) == (Outcome.REVISED, denied)
    assert llm.calls[0]["system"] == REVISE
    assert reflection.rounds[0].critic_issues is None


def test_reviser_gets_the_issues_and_only_the_submit_tool():
    llm = FakeLLM(review(BY_FRIDAY), revise(GOOD), review())

    run_reflect(llm, BAD)

    sent = llm.calls[1]
    content = json.loads(sent["messages"][0]["content"])
    assert content["issues"] == ['"by Friday": No tool result mentions a date.']
    assert content["draft"] == BAD
    assert sent["tools"] == [SUBMIT_TOOL]


def test_second_revision_starts_from_the_first():
    better = draft(message_to_employee="Approved. It should arrive by Friday.")
    llm = FakeLLM(review(BY_FRIDAY), revise(better), review(BY_FRIDAY), revise(GOOD), review())

    reflection = run_reflect(llm, BAD)

    assert json.loads(llm.calls[3]["messages"][0]["content"])["draft"] == better
    assert (reflection.outcome, reflection.decision) == (Outcome.REVISED, GOOD)


def test_still_failing_after_two_revisions_escalates():
    mcp = FakeMCP()
    llm = FakeLLM(review(BY_FRIDAY), revise(BAD), review(BY_FRIDAY), revise(BAD), review(BY_FRIDAY))

    reflection = run_reflect(llm, BAD, mcp=mcp)

    assert reflection.outcome == Outcome.ESCALATED
    assert len(reflection.rounds) == 3
    assert llm.replies == []
    assert (reflection.decision["decision"], reflection.decision["reason_code"]) == (
        "escalate",
        "REFLECTION_FAILED",
    )
    [(name, arguments)] = mcp.calls
    assert name == "flag_for_human_review"
    assert "by Friday" in arguments["reason"]


def test_max_revisions_can_be_changed():
    reflection = run_reflect(FakeLLM(review(BY_FRIDAY)), BAD, max_revisions=0)

    assert reflection.outcome == Outcome.ESCALATED


def test_failing_draft_that_was_escalated_keeps_its_record():
    mcp = FakeMCP()
    bad = escalation(message_to_employee="ESC-1: you will surely get it.")
    llm = FakeLLM(review(BY_FRIDAY))

    reflection = run_reflect(llm, bad, [flagged()], mcp=mcp, max_revisions=0)

    assert mcp.calls == []
    assert reflection.decision["escalation_id"] == "ESC-1"
    assert reflection.decision["reason_code"] == "EXCEPTION_REQUEST"
    assert "surely" not in reflection.decision["message_to_employee"]


@pytest.mark.parametrize(
    ("before", "after", "steps"),
    [
        pytest.param(BAD, escalation(escalation_id=None), [eligibility()], id="to escalate"),
        pytest.param(escalation(), draft(), [eligibility()], id="from escalate"),
    ],
)
def test_revision_cannot_switch_to_or_from_escalate(before, after, steps):
    llm = FakeLLM(*([review(BY_FRIDAY)] if before is BAD else []), revise(after))

    reflection = run_reflect(llm, before, steps)

    assert reflection.outcome == Outcome.ESCALATED
    assert reflection.decision["reason_code"] == "REFLECTION_FAILED"
    assert llm.replies == []


def test_revision_may_switch_between_approve_and_deny():
    approved = draft(reason_code="WITHIN_LIMIT")
    llm = FakeLLM(revise(approved), review())

    reflection = run_reflect(llm, draft(decision="deny", reason_code="REFRESH_NOT_DUE"))

    assert (reflection.outcome, reflection.decision) == (Outcome.REVISED, approved)


def test_critic_without_a_verdict_escalates():
    reflection = run_reflect(FakeLLM(reply(stop="refusal")))

    assert reflection.outcome == Outcome.ESCALATED


def test_reviser_without_a_draft_escalates():
    llm = FakeLLM(review(BY_FRIDAY), reply(thought("Sorry"), stop="end_turn"))

    assert run_reflect(llm, BAD).outcome == Outcome.ESCALATED


def test_decision_made_by_code_is_not_reviewed():
    llm = FakeLLM()
    made_by_code = escalation(reason_code="AGENT_STEP_LIMIT")

    reflection = run_reflect(llm, made_by_code, stop="step_limit")

    assert (reflection.outcome, reflection.decision) == (Outcome.SKIPPED, made_by_code)
    assert llm.calls == []


def test_every_round_is_reported_as_it_happens():
    seen = []
    llm = FakeLLM(review(BY_FRIDAY), revise(GOOD), review())

    reflection = run_reflect(llm, BAD, on_round=seen.append)

    assert seen == reflection.rounds


def test_fallback_escalation_is_saved_by_the_real_server(server_db, monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")

    async def go():
        async with connect() as mcp:
            result = AgentResult(BAD, "submitted", [eligibility()])
            return await reflect(FakeLLM(review(BY_FRIDAY)), mcp, REQUEST, result, max_revisions=0)

    reflection = asyncio.run(go())

    with closing(sqlite3.connect(server_db)) as conn:
        rows = conn.execute("SELECT id, employee_id, reason_code FROM escalations").fetchall()
    assert rows == [(reflection.decision["escalation_id"], "E001", "REFLECTION_FAILED")]


# ---------------------------------------------------------------- trace


def test_format_round_shows_the_draft_and_the_critic_issues():
    current = Round(BAD, [], ['"by Friday": No tool result mentions a date.'])

    assert format_round(1, current) == (
        "Reflection: draft 1\n"
        "  Decision:    APPROVE (WITHIN_LIMIT)\n"
        "  Message:     Approved. You will have it by Friday.\n"
        "  Code check:  pass\n"
        "  Critic:      1 issue\n"
        '    - "by Friday": No tool result mentions a date.'
    )


@pytest.mark.parametrize(
    ("current", "code_line", "critic_line"),
    [
        (Round(GOOD, [], []), "Code check:  pass", "Critic:      pass"),
        (Round(GOOD, ["a", "b"]), "Code check:  2 issues", "Critic:      not run"),
        (Round(GOOD, [], None), "Code check:  pass", "Critic:      no verdict"),
    ],
)
def test_format_round_shows_each_check(current, code_line, critic_line):
    lines = [line.strip() for line in format_round(1, current).splitlines()]

    assert code_line in lines
    assert critic_line in lines


@pytest.mark.parametrize(
    ("outcome", "rounds", "reason", "expected"),
    [
        (Outcome.CONFIRMED, 1, None, "Reflection: CONFIRMED (0 revisions)"),
        (Outcome.REVISED, 2, None, "Reflection: REVISED (1 revision)"),
        (Outcome.ESCALATED, 3, "Why.", "Reflection: ESCALATED (2 revisions): Why."),
        (Outcome.SKIPPED, 0, None, "Reflection: SKIPPED (decision made by code)"),
    ],
)
def test_format_outcome(outcome, rounds, reason, expected):
    reflection = Reflection(GOOD, outcome, [Round(GOOD, [])] * rounds, reason)

    assert format_outcome(reflection) == expected


def test_cli_prints_the_reflected_decision(server_db, monkeypatch, capsys):
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    eligible = call("check_request_eligibility", {"employee_id": "E001", "item": "monitor"})
    llm = FakeLLM(reply(eligible), revise(BAD), review(BY_FRIDAY), revise(GOOD), review())
    monkeypatch.setattr("equipment.agent.cli.FoundryClient.from_env", lambda: llm)

    code = asyncio.run(run(REQUEST))

    output = capsys.readouterr().out
    final = output[output.index("Reflection: REVISED") :]
    assert code == 0
    assert "Reflection: draft 1" in output and "Reflection: draft 2" in output
    assert GOOD["message_to_employee"] in final
    assert BAD["message_to_employee"] not in final
