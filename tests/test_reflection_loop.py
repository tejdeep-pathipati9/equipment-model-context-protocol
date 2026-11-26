"""A rejected draft goes back to the ReAct loop. Scripted fake LLM, real MCP server over stdio."""

import asyncio
import json
import sqlite3
from contextlib import closing
from functools import partial

import pytest
from fakes import FakeLLM, accept, call, reply, thought

from equipment.agent.mcp_client import connect
from equipment.agent.prompts import CRITIC, REJECTED
from equipment.agent.react import run_agent
from equipment.agent.reflection import critique
from equipment.agent.types import Draft, Outcome, Proposal, Request, Stop

GOOD = {
    "decision": "approve",
    "reason_code": "WITHIN_LIMIT",
    "item": "monitor",
    "message_to_employee": "Approved. I cannot confirm a delivery date.",
    "escalation_id": None,
}
BAD = GOOD | {"message_to_employee": "Approved. You will have it by Friday."}
REQUEST = Request("E001", "Can I get a second monitor by Friday?")
ISSUE = '"by Friday": No tool result mentions a date.'


def submit(draft, id="s1"):
    return reply(call("submit_decision", draft, id))


def look_up(item="monitor", employee_id="E001"):
    arguments = {"employee_id": employee_id, "item": item}
    return reply(call("check_request_eligibility", arguments, "c1"))


def flag(reason_code="EXCEPTION_REQUEST"):
    arguments = {
        "employee_id": "E001",
        "request": "broken",
        "reason": "Breakage cannot be verified.",
        "reason_code": reason_code,
    }
    return reply(call("flag_for_human_review", arguments, "f1"))


class FakeCritic:
    """Returns scripted issue lists in order; [] accepts, None gives no verdict."""

    def __init__(self, *verdicts):
        self.verdicts = list(verdicts)
        self.calls = []

    async def __call__(self, request, draft, steps):
        self.calls.append((request, draft, list(steps)))
        return self.verdicts.pop(0)


@pytest.fixture
def agent(server_db, monkeypatch):
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")

    def run(llm, critic, **options):
        async def go():
            async with connect() as mcp:
                return await run_agent(llm, mcp, REQUEST, critic=critic, **options)

        return asyncio.run(go())

    return run


def saved(db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute("SELECT id, reason_code, reason FROM escalations").fetchall()


# ---------------------------------------------------------------- the loop and a critic


def test_accepted_first_draft_is_confirmed(agent):
    critic = FakeCritic([])

    result = agent(FakeLLM(look_up(), submit(GOOD)), critic)

    assert (result.stop, result.outcome, result.decision) == (
        Stop.SUBMITTED,
        Outcome.CONFIRMED,
        Draft(**GOOD),
    )
    [(request, draft, steps)] = critic.calls
    assert (request, draft) == (REQUEST, Draft(**GOOD))
    assert [step.action for step in steps] == ["check_request_eligibility"]


def test_rejected_draft_goes_back_to_the_model_with_its_issues(agent):
    llm = FakeLLM(look_up(), submit(BAD, "s1"), submit(GOOD, "s2"))

    result = agent(llm, FakeCritic([ISSUE], []))

    [rejection] = llm.calls[2]["messages"][-1]["content"]
    assert (rejection["tool_use_id"], rejection["is_error"]) == ("s1", True)
    issues, rules = rejection["content"]
    assert json.loads(issues["text"]) == {"accepted": False, "issues": [ISSUE]}
    assert rules["text"] == REJECTED
    assert (result.outcome, result.decision) == (Outcome.REVISED, Draft(**GOOD))


def test_wording_fix_needs_one_more_turn_and_no_lookup(agent):
    llm = FakeLLM(look_up(), submit(BAD), submit(GOOD))

    result = agent(llm, FakeCritic([ISSUE], []))

    assert len(llm.calls) == 3
    assert [step.action for step in result.steps] == [
        "check_request_eligibility",
        "submit_decision",
        "submit_decision",
    ]


def test_rejected_model_keeps_its_tools(agent):
    llm = FakeLLM(look_up(), submit(BAD), submit(GOOD))

    agent(llm, FakeCritic([ISSUE], []))

    assert "check_request_eligibility" in [tool["name"] for tool in llm.calls[2]["tools"]]


def test_rejection_is_a_step_in_the_trace(agent):
    llm = FakeLLM(look_up(), submit(BAD), submit(GOOD))

    result = agent(llm, FakeCritic([ISSUE], []))

    _, rejected, accepted = result.steps
    assert rejected.observation == {"accepted": False, "issues": [ISSUE]}
    assert rejected.is_error is True
    assert accepted.observation == {"accepted": True}


def test_every_review_is_kept_and_reported(agent):
    seen = []
    llm = FakeLLM(look_up(), submit(BAD), submit(GOOD))

    result = agent(llm, FakeCritic([ISSUE], []), on_review=seen.append)

    assert [review.draft for review in result.reviews] == [BAD, GOOD]
    assert seen == result.reviews


def test_third_rejected_draft_escalates(agent, server_db):
    llm = FakeLLM(look_up(), submit(BAD), submit(BAD), submit(BAD))

    result = agent(llm, FakeCritic([ISSUE], [ISSUE], [ISSUE]))

    [(id, reason_code, reason)] = saved(server_db)
    assert (result.stop, result.outcome) == (Stop.REFLECTION_FAILED, Outcome.ESCALATED)
    assert (result.decision.escalation_id, reason_code) == (id, "REFLECTION_FAILED")
    assert "by Friday" in reason
    assert llm.replies == []


def test_max_revisions_can_be_changed(agent):
    result = agent(FakeLLM(look_up(), submit(BAD)), FakeCritic([ISSUE]), max_revisions=0)

    assert result.stop == Stop.REFLECTION_FAILED


def test_no_verdict_escalates_at_once(agent, server_db):
    result = agent(FakeLLM(look_up(), submit(GOOD)), FakeCritic(None))

    assert result.stop == Stop.REFLECTION_FAILED
    assert "no verdict" in saved(server_db)[0][2]


def test_failed_reflection_keeps_a_saved_escalation(agent, server_db):
    llm = FakeLLM(flag("VAGUE_REQUEST"), submit(BAD))

    result = agent(llm, FakeCritic([]), max_revisions=0)

    [(id, reason_code, _)] = saved(server_db)
    assert result.decision.escalation_id == id
    assert result.decision.reason_code == reason_code == "VAGUE_REQUEST"
    assert "Friday" not in result.decision.message_to_employee


def test_revisions_share_the_step_limit(agent):
    llm = FakeLLM(submit(BAD), look_up(), look_up())

    result = agent(llm, FakeCritic([ISSUE]), max_steps=3)

    assert (result.stop, result.outcome) == (Stop.STEP_LIMIT, Outcome.ESCALATED)
    assert result.decision.reason_code == "AGENT_STEP_LIMIT"


def test_refusal_after_a_rejection_escalates(agent):
    llm = FakeLLM(submit(BAD), reply(stop="refusal"))

    result = agent(llm, FakeCritic([ISSUE]))

    assert (result.stop, result.outcome) == (Stop.REFUSED, Outcome.ESCALATED)


def test_decision_made_by_code_is_not_reviewed(agent):
    critic = FakeCritic()

    result = agent(FakeLLM(reply(stop="refusal")), critic)

    assert result.outcome == Outcome.SKIPPED
    assert critic.calls == []


def test_code_check_runs_before_the_critic(agent):
    critic = FakeCritic([])
    llm = FakeLLM(submit(GOOD, "s1"), look_up(), submit(GOOD, "s2"))

    result = agent(llm, critic)

    assert "no eligibility result" in result.steps[0].observation["issues"][0]
    assert (result.outcome, result.decision) == (Outcome.REVISED, Draft(**GOOD))
    assert len(critic.calls) == 1  # the critic never saw the draft the code check rejected


def test_code_check_runs_even_when_the_critic_accepts_everything(agent):
    llm = FakeLLM(submit(GOOD, "s1"), look_up(), submit(GOOD, "s2"))

    result = agent(llm, accept)

    assert "no eligibility result" in result.steps[0].observation["issues"][0]
    assert (result.outcome, result.decision) == (Outcome.REVISED, Draft(**GOOD))


# ---------------------------------------------------------------- escalations are reviewed first

REFUSE = reply(stop="refusal")  # ends the run, so a test can look at what was saved


def test_wrong_escalation_is_not_saved(agent, server_db):
    llm = FakeLLM(look_up(), flag("EXCEPTION_REQUEST"), REFUSE)

    agent(llm, FakeCritic())

    assert [code for _, code, _ in saved(server_db)] == ["AGENT_REFUSED"]


def test_rejected_escalation_goes_back_to_the_model(agent):
    llm = FakeLLM(look_up(), flag("EXCEPTION_REQUEST"), REFUSE)

    result = agent(llm, FakeCritic())

    assert "WITHIN_LIMIT" in result.steps[1].observation["issues"][0]


def test_model_can_approve_after_its_escalation_is_rejected(agent):
    llm = FakeLLM(look_up(), flag("EXCEPTION_REQUEST"), submit(GOOD))

    result = agent(llm, FakeCritic([]))

    assert (result.outcome, result.decision) == (Outcome.REVISED, Draft(**GOOD))


def test_critic_can_stop_an_escalation_before_it_is_saved(agent, server_db):
    llm = FakeLLM(look_up(), flag("VAGUE_REQUEST"), REFUSE)

    agent(llm, FakeCritic([ISSUE]))

    assert [code for _, code, _ in saved(server_db)] == ["AGENT_REFUSED"]


def test_reviewed_escalation_is_saved(agent, server_db):
    llm = FakeLLM(look_up(), flag("VAGUE_REQUEST"), REFUSE)

    agent(llm, FakeCritic([]))

    assert [code for _, code, _ in saved(server_db)] == ["VAGUE_REQUEST"]


def test_critic_sees_the_proposed_escalation(agent):
    critic = FakeCritic([])

    agent(FakeLLM(look_up(), flag("VAGUE_REQUEST"), REFUSE), critic)

    [(_, proposal, _)] = critic.calls
    assert proposal == Proposal(reason_code="VAGUE_REQUEST", reason="Breakage cannot be verified.")


def test_escalation_is_saved_when_the_critic_gives_no_verdict(agent, server_db):
    llm = FakeLLM(look_up(), flag("VAGUE_REQUEST"), REFUSE)

    agent(llm, FakeCritic(None))

    assert [code for _, code, _ in saved(server_db)] == ["VAGUE_REQUEST"]


def test_no_verdict_on_an_escalation_is_not_a_revision(agent):
    llm = FakeLLM(look_up(), flag("VAGUE_REQUEST"), submit(GOOD), REFUSE)

    result = agent(llm, FakeCritic(None), max_revisions=1)

    assert result.stop == Stop.REFUSED  # only the approval was rejected


def test_rejected_escalation_counts_as_a_revision(agent):
    llm = FakeLLM(look_up(), flag("EXCEPTION_REQUEST"))

    result = agent(llm, FakeCritic(), max_revisions=0)

    assert result.stop == Stop.REFLECTION_FAILED


def test_accepted_escalation_does_not_use_up_a_revision(agent):
    llm = FakeLLM(look_up(), flag("VAGUE_REQUEST"), submit(GOOD), REFUSE)

    result = agent(llm, FakeCritic([]), max_revisions=1)

    assert result.stop == Stop.REFUSED  # one rejection (the approval) is within the limit


# ---------------------------------------------------------------- invalid drafts

INVALID = [
    pytest.param(GOOD | {"decision": "maybe"}, "decision", id="unknown decision"),
    pytest.param(GOOD | {"reason_code": "BECAUSE"}, "reason_code", id="unknown reason code"),
    pytest.param(GOOD | {"message_to_employee": 123}, "message_to_employee", id="not text"),
    pytest.param(GOOD | {"message_to_employee": "  "}, "message_to_employee", id="empty message"),
    pytest.param(GOOD | {"priority": "high"}, "priority", id="extra field"),
    pytest.param({"decision": "approve"}, "reason_code", id="cut off"),
]


@pytest.mark.parametrize(("invalid", "field"), INVALID)
def test_invalid_draft_goes_back_to_the_model(agent, invalid, field):
    critic = FakeCritic([])

    result = agent(FakeLLM(look_up(), submit(invalid, "s1"), submit(GOOD, "s2")), critic)

    assert any(field in issue for issue in result.steps[1].observation["issues"])
    assert result.reviews[0].draft == invalid
    assert (result.outcome, result.decision) == (Outcome.REVISED, Draft(**GOOD))
    assert len(critic.calls) == 1  # the invalid draft never reaches the critic


def test_always_invalid_drafts_escalate(agent):
    llm = FakeLLM(*[submit({"decision": "approve"})] * 3)

    assert agent(llm, FakeCritic()).stop == Stop.REFLECTION_FAILED


# ---------------------------------------------------------------- with the real review


def critic_says(*issues):
    return reply(call("submit_review", {"issues": list(issues)}))


def run_reviewed(agent, *replies):
    llm = FakeLLM(*replies)
    return agent(llm, partial(critique, llm)), llm


def test_missing_lookup_is_done_before_the_draft_is_accepted(agent):
    result, _ = run_reviewed(
        agent,
        reply(thought("Looks fine."), call("submit_decision", GOOD, "s1")),
        reply(thought("I must check eligibility first."), *look_up().content),
        submit(GOOD, "s2"),
        critic_says(),
    )

    assert [step.action for step in result.steps] == [
        "submit_decision",
        "check_request_eligibility",
        "submit_decision",
    ]
    assert "no eligibility result" in result.steps[0].observation["issues"][0]
    assert (result.outcome, result.decision) == (Outcome.REVISED, Draft(**GOOD))


def test_approval_can_become_a_real_escalation(agent, server_db):
    """The reviewer says escalate; the model flags it itself, with the right reason code."""
    issue = {"quote": "approve", "problem": "Rule 4 requires escalation."}
    llm = FakeLLM(look_up(), submit(GOOD, "s1"), critic_says(issue), flag("VAGUE_REQUEST"))

    def submit_saved_id():
        [(id, _, _)] = saved(server_db)
        return submit(
            GOOD
            | {
                "decision": "escalate",
                "reason_code": "VAGUE_REQUEST",
                "message_to_employee": f"Passed to a human reviewer ({id}).",
                "escalation_id": id,
            },
            "s2",
        )

    async def create(**sent):
        llm.calls.append(sent)
        if llm.replies:
            return llm.replies.pop(0)
        return critic_says() if sent["system"] == CRITIC else submit_saved_id()

    llm.create = create
    result = agent(llm, partial(critique, llm))

    assert (result.stop, result.outcome) == (Stop.SUBMITTED, Outcome.REVISED)
    assert (result.decision.decision, result.decision.reason_code) == (
        "escalate",
        "VAGUE_REQUEST",
    )
    assert len(saved(server_db)) == 1


def test_saved_escalation_cannot_become_an_approval(agent, server_db):
    result, llm = run_reviewed(
        agent,
        look_up(),
        flag("VAGUE_REQUEST"),
        critic_says(),
        submit(GOOD, "s1"),
        submit(GOOD, "s2"),
        submit(GOOD, "s3"),
    )

    assert result.decision.decision == "escalate"
    assert len(saved(server_db)) == 1
    assert all("already escalated" in review.issues[0] for review in result.reviews[1:])
