"""Offline checks for the dataset and paired capture. No model or network requests."""

import asyncio
import json
from collections import Counter
from pathlib import Path

import pytest
from anthropic.types import Usage
from evals.comparison import Comparison, matches, usage_totals
from fakes import FakeLLM, call, reply

from equipment import tools
from equipment.agent.checks import check_draft
from equipment.agent.mcp_client import connect
from equipment.agent.prompts import CRITIC, SYSTEM
from equipment.agent.react import run_agent
from equipment.agent.types import AgentResult, Draft, FinalDecision, Request, Step, Stop

CASES = json.loads(Path("scenarios/comparison.json").read_text())


def test_comparison_has_ten_unique_cases():
    assert len({case["id"] for case in CASES}) == len(CASES) == 10


def test_comparison_balances_automatic_and_escalated_decisions():
    counts = Counter(case["expected"]["decision"] for case in CASES)
    assert counts == {"approve": 3, "deny": 2, "escalate": 5}


def test_comparison_includes_the_four_required_scenarios():
    demos = json.loads(Path("scenarios/demo.json").read_text())
    for demo in demos:
        case = next(case for case in CASES if case["text"] == demo["text"])
        assert case["employee_id"] == demo["employee_id"]
        assert matches(case["expected"], demo["expected"])


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_expected_decision_passes_code_checks_against_seeded_records(case, repo, as_of):
    expected = case["expected"]
    item = expected.get("item")
    request = Request(case["employee_id"], case["text"])
    steps = [
        Step(
            "",
            "get_employee_info",
            {},
            tools.get_employee_info(repo, request.employee_id, as_of),
            False,
        )
    ]
    if item:
        verdict = tools.check_request_eligibility(repo, request.employee_id, item, as_of)
        steps.append(Step("", "check_request_eligibility", {}, verdict, False))
    draft = Draft(
        **expected,
        **({"item": None} if not item else {}),
        extra_unit=case["id"] == "05-additional-laptop",
        message_to_employee="Test draft.",
        escalation_reason="Test note." if expected["decision"] == "escalate" else None,
    )
    assert check_draft(draft, steps) == []


@pytest.fixture
def capture(monkeypatch):
    async def critic(*args):
        return ["Unsupported promise"]

    monkeypatch.setattr("evals.comparison.critique", critic)
    return Comparison(FakeLLM())


def proposed(message="Approved."):
    return Draft(
        decision="approve",
        reason_code="WITHIN_LIMIT",
        item="monitor",
        extra_unit=False,
        message_to_employee=message,
        escalation_reason=None,
    )


def test_first_draft_is_preserved_after_a_revision(capture):
    request = Request("E001", "Can I get a second monitor?")
    first = proposed("You will have it by Friday.")
    asyncio.run(capture.review(request, first, []))
    asyncio.run(capture.review(request, proposed(), []))
    assert capture.first_draft == first.model_dump(mode="json")


def test_usage_is_split_before_and_after_the_first_review(capture):
    replies = [reply(), reply()]
    for response in replies:
        response.usage = Usage(input_tokens=20, output_tokens=5, cache_read_input_tokens=10)
    capture.llm = FakeLLM(*replies)

    async def go():
        await capture.create(system=SYSTEM, messages=[], tools=[])
        await capture.review(Request("E001", "A monitor please."), proposed(), [])
        await capture.create(system=CRITIC, messages=[], tools=[])

    asyncio.run(go())
    assert capture.first_call_count == 1
    assert usage_totals(capture.calls) == {
        "input_tokens": 40,
        "output_tokens": 10,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 20,
    }


def test_no_valid_draft_is_reported_as_a_missing_baseline(capture):
    final = FinalDecision(
        decision="escalate",
        reason_code="AGENT_STEP_LIMIT",
        item=None,
        message_to_employee="Sent for review.",
        escalation_id="ESC-test",
    )
    row = capture.report(CASES[0], AgentResult(final, Stop.STEP_LIMIT))
    assert (row["baseline_available"], row["before_correct"]) == (False, False)


def test_report_does_not_count_a_regression_as_an_improvement(capture):
    capture.first_draft = proposed().model_dump(mode="json")
    final = FinalDecision(
        decision="deny",
        reason_code="ROLE_NOT_ELIGIBLE",
        item="monitor",
        message_to_employee="Denied.",
    )
    row = capture.report(CASES[0], AgentResult(final, Stop.SUBMITTED))
    assert row["decision_change"] == "regressed"


def test_pair_starts_after_code_checks_and_preserves_the_unreviewed_answer(server_db, monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    first = proposed("Approved. You will have it by Friday.")
    replies = [
        reply(call("submit_decision", proposed("Unverified guess.").model_dump(mode="json"))),
        reply(call("check_request_eligibility", {"employee_id": "E001", "item": "monitor"})),
        reply(call("submit_decision", first.model_dump(mode="json"))),
        reply(
            call("submit_review", {"issues": [{"quote": "Friday", "problem": "No delivery date."}]})
        ),
        reply(call("submit_decision", proposed().model_dump(mode="json"))),
        reply(call("submit_review", {"issues": []})),
    ]
    for response in replies:
        response.usage = Usage(input_tokens=20, output_tokens=5)
    comparison = Comparison(FakeLLM(*replies))

    async def go():
        async with connect() as mcp:
            return await run_agent(
                comparison, mcp, Request("E001", CASES[0]["text"]), comparison.review
            )

    result = asyncio.run(go())
    row = comparison.report(CASES[0], result)

    assert row["first_valid_draft"] == first.model_dump(mode="json")
    assert (row["before_call_count"], len(row["calls"])) == (3, 6)
    assert row["before_usage"]["input_tokens"] == 60
    assert row["total_usage"]["input_tokens"] == 120
    assert row["final_decision"]["message_to_employee"] == "Approved."
