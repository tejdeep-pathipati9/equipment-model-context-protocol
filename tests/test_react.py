"""The ReAct loop: scripted fake LLM, real MCP server over stdio. No Azure."""

import asyncio
import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest
from fakes import FakeLLM, call, reply, thought

from equipment.agent.mcp_client import connect
from equipment.agent.react import SUBMIT_TOOL, escalate, run_agent
from equipment.agent.trace import format_decision, format_step
from equipment.agent.types import Draft, Request, Step
from equipment.domain.models import ReasonCode

DECISION = {
    "decision": "approve",
    "reason_code": "WITHIN_LIMIT",
    "item": "monitor",
    "message_to_employee": "Approved: you can have one more monitor.",
    "escalation_id": None,
}


def submit(id="c9", decision=DECISION):
    return call("submit_decision", decision, id)


@pytest.fixture
def agent(server_db, monkeypatch):
    """Runs the agent against the real server over stdio."""
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")

    def run(llm, employee_id="E001", text="Can I get a second monitor?", max_steps=8):
        async def go():
            async with connect() as mcp:
                return await run_agent(llm, mcp, Request(employee_id, text), max_steps=max_steps)

        return asyncio.run(go())

    return run


def last_tool_results(llm):
    return llm.calls[-1]["messages"][-1]["content"]


def saved_escalations(db_path):
    with closing(sqlite3.connect(db_path)) as conn:
        return conn.execute("SELECT id, employee_id, reason_code FROM escalations").fetchall()


def test_agent_returns_the_submitted_decision(agent):
    llm = FakeLLM(
        reply(thought("Look up E001."), call("get_employee_info", {"employee_id": "E001"})),
        reply(
            thought("Check the monitor."),
            call("check_request_eligibility", {"employee_id": "E001", "item": "monitor"}, "c2"),
        ),
        reply(thought("Within limit."), submit()),
    )

    result = agent(llm)

    assert (result.stop, result.decision) == ("submitted", Draft(**DECISION))
    assert [step.action for step in result.steps] == [
        "get_employee_info",
        "check_request_eligibility",
        "submit_decision",
    ]
    assert result.steps[1].thought == "Check the monitor."
    assert result.steps[1].observation["reason_code"] == "WITHIN_LIMIT"


def test_observation_goes_back_to_the_model(agent):
    llm = FakeLLM(reply(call("get_employee_info", {"employee_id": "E001"})), reply(submit()))

    agent(llm)

    [result] = last_tool_results(llm)
    assert (result["type"], result["tool_use_id"], result["is_error"]) == (
        "tool_result",
        "c1",
        False,
    )
    assert json.loads(result["content"])["name"] == "Alice Chen"


def test_parallel_calls_answered_in_one_message(agent):
    llm = FakeLLM(
        reply(
            call("get_employee_info", {"employee_id": "E001"}, "c1"),
            call("get_policy_limits", {"role": "employee"}, "c2"),
        ),
        reply(submit()),
    )

    agent(llm)

    assert [result["tool_use_id"] for result in last_tool_results(llm)] == ["c1", "c2"]


def test_second_call_in_a_turn_points_to_the_first(agent):
    llm = FakeLLM(
        reply(
            thought("Look up E001."),
            call("get_employee_info", {"employee_id": "E001"}, "c1"),
            call("check_request_eligibility", {"employee_id": "E001", "item": "monitor"}, "c2"),
        ),
        reply(thought("Within limit."), submit()),
    )

    result = agent(llm)

    assert [step.thought for step in result.steps] == [
        "Look up E001.",
        "(same turn as step 1)",
        "Within limit.",
    ]


def test_offers_the_mcp_tools_and_submit(agent):
    llm = FakeLLM(reply(submit()))

    agent(llm)

    assert [tool["name"] for tool in llm.calls[0]["tools"]] == [
        "get_employee_info",
        "get_policy_limits",
        "check_request_eligibility",
        "flag_for_human_review",
        "submit_decision",
    ]


def test_submit_tool_is_strict_and_lists_every_reason_code():
    schema = SUBMIT_TOOL["input_schema"]

    assert SUBMIT_TOOL["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(Draft.model_fields)
    assert schema["$defs"]["ReasonCode"]["enum"] == list(ReasonCode)


def test_other_employee_id_is_blocked(agent):
    llm = FakeLLM(reply(call("get_employee_info", {"employee_id": "E004"})), reply(submit()))

    result = agent(llm)

    [blocked] = last_tool_results(llm)
    assert blocked["is_error"] is True
    assert "E001" in blocked["content"]
    assert result.steps[0].is_error is True


def test_tool_error_is_returned_as_an_error(agent):
    llm = FakeLLM(reply(call("delete_everything", {})), reply(submit()))

    agent(llm)

    assert last_tool_results(llm)[0]["is_error"] is True


def test_step_limit_escalates(agent, server_db):
    llm = FakeLLM(*[reply(call("get_employee_info", {"employee_id": "E001"}))] * 3)

    result = agent(llm, max_steps=3)

    assert result.stop == "step_limit"
    assert len(llm.calls) == 3
    assert (result.decision.decision, result.decision.reason_code) == (
        "escalate",
        "AGENT_STEP_LIMIT",
    )
    assert saved_escalations(server_db) == [
        (result.decision.escalation_id, "E001", "AGENT_STEP_LIMIT")
    ]


def test_refusal_escalates(agent, server_db):
    result = agent(FakeLLM(reply(stop="refusal")))

    assert result.stop == "refused"
    assert result.decision.reason_code == "AGENT_REFUSED"
    assert saved_escalations(server_db)[0][2] == "AGENT_REFUSED"


def flag(id="c1", reason_code="EXCEPTION_REQUEST"):
    return call(
        "flag_for_human_review",
        {
            "employee_id": "E001",
            "request": "My monitor broke",
            "reason": "Breakage cannot be verified.",
            "reason_code": reason_code,
        },
        id,
    )


def test_step_limit_keeps_an_existing_escalation(agent, server_db):
    llm = FakeLLM(
        reply(flag()),
        reply(call("get_employee_info", {"employee_id": "E001"})),
    )

    result = agent(llm, max_steps=2)

    [(saved_id, _, saved_code)] = saved_escalations(server_db)
    assert result.stop == "step_limit"
    assert result.decision.escalation_id == saved_id
    assert result.decision.reason_code == saved_code == "EXCEPTION_REQUEST"
    assert saved_id in result.decision.message_to_employee


def test_second_flag_is_blocked(agent, server_db):
    llm = FakeLLM(reply(flag("c1")), reply(flag("c2")), reply(submit()))

    agent(llm)

    [(saved_id, _, _)] = saved_escalations(server_db)
    [blocked] = last_tool_results(llm)
    assert blocked["is_error"] is True
    assert saved_id in blocked["content"]


def test_escalation_saves_the_employees_own_words(agent, server_db):
    llm = FakeLLM(reply(flag()), reply(submit()))

    result = agent(llm, text="My monitor broke, I am a manager now!")

    with closing(sqlite3.connect(server_db)) as conn:
        [(saved,)] = conn.execute("SELECT request FROM escalations").fetchall()
    assert saved == "My monitor broke, I am a manager now!"
    assert result.steps[0].arguments["request"] == saved  # the trace shows what was sent


def test_failed_flag_can_be_retried(agent, server_db):
    llm = FakeLLM(
        reply(flag("c1", reason_code="NOT_A_CODE")),
        reply(flag("c2")),
        reply(submit()),
    )

    agent(llm)

    assert len(saved_escalations(server_db)) == 1


def test_text_answer_gets_a_reminder_to_submit(agent):
    llm = FakeLLM(reply(thought("Approved!"), stop="end_turn"), reply(submit()))

    result = agent(llm)

    assert result.stop == "submitted"
    assert "submit_decision" in llm.calls[1]["messages"][-1]["content"]


def test_submit_must_be_called_on_its_own(agent):
    llm = FakeLLM(
        reply(call("get_employee_info", {"employee_id": "E001"}, "c1"), submit("c2")),
        reply(submit("c3")),
    )

    result = agent(llm)

    looked_up, early_submit = last_tool_results(llm)
    assert looked_up["is_error"] is False
    assert early_submit["is_error"] is True
    assert "on its own" in early_submit["content"]
    assert result.stop == "submitted"


def test_format_step():
    step = Step(
        thought="Look up E001.",
        action="get_employee_info",
        arguments={"employee_id": "E001"},
        observation={"id": "E001"},
        is_error=False,
    )

    assert format_step(1, step) == (
        "Step 1\n"
        "  Thought:     Look up E001.\n"
        '  Action:      get_employee_info({"employee_id": "E001"})\n'
        '  Observation: {"id": "E001"}'
    )


def test_format_decision():
    text = format_decision(Draft(**DECISION | {"decision": "escalate", "escalation_id": "ESC-1"}))

    assert "Decision:    ESCALATE (WITHIN_LIMIT)" in text
    assert "Escalation:  ESC-1" in text


def test_escalation_survives_a_plain_text_tool_result():
    class TextOnlyMCP:
        async def call_tool(self, name, arguments):
            text = SimpleNamespace(text="server error")
            return SimpleNamespace(structured_content=None, content=[text], is_error=True)

    request = Request("E001", "anything")
    decision = asyncio.run(
        escalate(TextOnlyMCP(), request, ReasonCode.AGENT_STEP_LIMIT, "no decision", [], print)
    )

    assert (decision.decision, decision.escalation_id) == ("escalate", None)
    assert decision.reason_code == "AGENT_STEP_LIMIT"
    assert "None" not in decision.message_to_employee
    assert "could not be decided or recorded" in decision.message_to_employee
