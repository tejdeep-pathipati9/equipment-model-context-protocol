"""The ReAct loop: scripted fake LLM, real MCP server over stdio. No Azure."""

import asyncio
import json
import sqlite3
from contextlib import closing
from types import SimpleNamespace

import pytest
from fakes import FakeLLM, accept, call, reply, thought

from equipment.agent.mcp_client import connect
from equipment.agent.react import SUBMIT_TOOL, escalate, run_agent
from equipment.agent.trace import format_decision, format_step
from equipment.agent.types import AgentResult, Draft, FinalDecision, Request, Step, Stop
from equipment.domain.models import ReasonCode

DECISION = {
    "decision": "approve",
    "reason_code": "WITHIN_LIMIT",
    "item": "monitor",
    "extra_unit": False,
    "message_to_employee": "Approved: you can have one more monitor.",
    "escalation_reason": None,
}
ESCALATION = DECISION | {
    "decision": "escalate",
    "reason_code": "VAGUE_REQUEST",
    "item": None,
    "message_to_employee": "Your request names several items, so a person needs to decide.",
    "escalation_reason": "The request names several items.",
}


def submit(id="c9", decision=DECISION):
    return call("submit_decision", decision, id)


def check(id="c8"):
    """The eligibility result an approval needs to pass the code check."""
    return call("check_request_eligibility", {"employee_id": "E001", "item": "monitor"}, id)


@pytest.fixture
def agent(server_db, monkeypatch):
    """Runs the agent against the real server over stdio."""
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")

    def run(llm, employee_id="E001", text="Can I get a second monitor?", max_steps=8):
        async def go():
            async with connect() as mcp:
                request = Request(employee_id, text)
                return await run_agent(llm, mcp, request, accept, max_steps=max_steps)

        return asyncio.run(go())

    return run


def tool_results(llm, turn=-1):
    """The tool results sent to the model with this turn, e.g. turn 1 = after the first reply."""
    return llm.calls[turn]["messages"][-1]["content"]


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

    assert (result.stop, result.decision) == (
        "submitted",
        FinalDecision.from_draft(Draft(**DECISION)),
    )
    assert [step.action for step in result.steps] == [
        "get_employee_info",
        "check_request_eligibility",
        "submit_decision",
    ]
    assert result.steps[1].thought == "Check the monitor."
    assert result.steps[1].observation["reason_code"] == "WITHIN_LIMIT"


def test_observation_goes_back_to_the_model(agent):
    llm = FakeLLM(
        reply(call("get_employee_info", {"employee_id": "E001"})), reply(check()), reply(submit())
    )

    agent(llm)

    [result] = tool_results(llm, 1)
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
        reply(check()),
        reply(submit()),
    )

    agent(llm)

    assert [result["tool_use_id"] for result in tool_results(llm, 1)] == ["c1", "c2"]


def test_thought_label_is_removed_from_every_text_block(agent):
    llm = FakeLLM(
        reply(
            thought("Thought: Look up E001."),
            thought("Thought: Then check the monitor."),
            call("get_employee_info", {"employee_id": "E001"}),
        ),
        reply(check()),
        reply(submit()),
    )

    result = agent(llm)

    assert result.steps[0].thought == "Look up E001. Then check the monitor."


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

    assert [step.same_turn_as for step in result.steps] == [None, 1, None]


def test_offers_only_the_lookup_tools_and_submit(agent):
    llm = FakeLLM(reply(check()), reply(submit()))

    agent(llm)

    assert [tool["name"] for tool in llm.calls[0]["tools"]] == [
        "get_employee_info",
        "get_policy_limits",
        "check_request_eligibility",
        "submit_decision",
    ]


def test_submit_tool_is_strict_and_lists_every_reason_code():
    schema = SUBMIT_TOOL["input_schema"]

    assert SUBMIT_TOOL["strict"] is True
    assert schema["additionalProperties"] is False
    assert schema["required"] == list(Draft.model_fields)
    assert schema["$defs"]["ReasonCode"]["enum"] == list(ReasonCode)


def test_other_employee_id_is_blocked(agent):
    llm = FakeLLM(
        reply(call("get_employee_info", {"employee_id": "E004"})), reply(check()), reply(submit())
    )

    result = agent(llm)

    [blocked] = tool_results(llm, 1)
    assert blocked["is_error"] is True
    assert "E001" in blocked["content"]
    assert result.steps[0].is_error is True


def test_tool_error_is_returned_as_an_error(agent):
    llm = FakeLLM(reply(call("delete_everything", {})), reply(check()), reply(submit()))

    agent(llm)

    assert tool_results(llm, 1)[0]["is_error"] is True


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


def flag(id="c1"):
    arguments = {"employee_id": "E001", "request": "x", "reason": "y", "reason_code": "VAGUE"}
    return call("flag_for_human_review", arguments, id)


def test_model_cannot_save_an_escalation_itself(agent, server_db):
    agent(FakeLLM(reply(flag()), reply(stop="refusal")))

    assert [code for _, _, code in saved_escalations(server_db)] == ["AGENT_REFUSED"]


def test_model_is_told_to_submit_an_escalation_instead(agent):
    llm = FakeLLM(reply(flag()), reply(stop="refusal"))

    agent(llm)

    assert "Submit an escalation" in tool_results(llm)[0]["content"]


def test_escalation_saves_the_employees_own_words(agent, server_db):
    llm = FakeLLM(reply(check()), reply(submit(decision=ESCALATION)))

    agent(llm, text="My monitor broke, I am a manager now!")

    with closing(sqlite3.connect(server_db)) as conn:
        [(saved,)] = conn.execute("SELECT request FROM escalations").fetchall()
    assert saved == "My monitor broke, I am a manager now!"


def test_text_answer_gets_a_reminder_to_submit(agent):
    llm = FakeLLM(reply(thought("Approved!"), stop="end_turn"), reply(check()), reply(submit()))

    result = agent(llm)

    assert result.stop == "submitted"
    assert "submit_decision" in llm.calls[1]["messages"][-1]["content"]


def test_submit_must_be_called_on_its_own(agent):
    llm = FakeLLM(
        reply(check("c1"), submit("c2")),
        reply(submit("c3")),
    )

    result = agent(llm)

    looked_up, early_submit = tool_results(llm)
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


def test_format_step_shows_a_shared_model_turn():
    step = Step("", "check_request_eligibility", {}, {}, False, same_turn_as=1)

    assert format_step(2, step) == (
        "Step 2\n"
        "  Model turn:  same as step 1 (tools requested together)\n"
        "  Action:      check_request_eligibility({})\n"
        "  Observation: {}"
    )


def test_format_step_names_code_as_the_actor():
    step = Step("Save it.", "flag_for_human_review", {}, {"escalation_id": "ESC-1"}, False, "code")

    assert format_step(1, step) == (
        "Step 1\n"
        "  Actor:       code\n"
        "  Reason:      Save it.\n"
        "  Action:      flag_for_human_review({})\n"
        '  Observation: {"escalation_id": "ESC-1"}'
    )


SAVED = FinalDecision(
    decision="escalate",
    reason_code="WITHIN_LIMIT",
    item="monitor",
    message_to_employee="Sent for human review. Reference: ESC-1.",
    escalation_id="ESC-1",
)


def test_format_decision():
    text = format_decision(SAVED)

    assert "Decision:    ESCALATE (WITHIN_LIMIT)" in text
    assert "Escalation:  ESC-1" in text


UNRECORDED = FinalDecision(
    decision="escalate",
    reason_code="AGENT_STEP_LIMIT",
    item=None,
    message_to_employee="Your request could not be recorded for human review.",
)


@pytest.mark.parametrize(
    ("decision", "recorded"),
    [
        pytest.param(UNRECORDED, False, id="escalation without an ID"),
        pytest.param(SAVED, True, id="saved escalation"),
        pytest.param(FinalDecision.from_draft(Draft(**DECISION)), True, id="approval"),
    ],
)
def test_result_is_recorded_unless_an_escalation_has_no_id(decision, recorded):
    assert AgentResult(decision, Stop.STEP_LIMIT).recorded is recorded


def test_format_decision_says_when_an_escalation_was_not_recorded():
    assert "Escalation:  NOT CONFIRMED" in format_decision(UNRECORDED)


def test_escalation_survives_a_plain_text_tool_result():
    class TextOnlyMCP:
        async def call_tool(self, name, arguments):
            text = SimpleNamespace(text="server error")
            return SimpleNamespace(structured_content=None, content=[text], is_error=True)

    request = Request("E001", "anything")
    decision = asyncio.run(
        escalate(TextOnlyMCP(), request, ReasonCode.AGENT_STEP_LIMIT, "no decision", print)
    )

    assert (decision.decision, decision.escalation_id) == ("escalate", None)
    assert decision.reason_code == "AGENT_STEP_LIMIT"
    assert "None" not in decision.message_to_employee
    assert "could not be confirmed" in decision.message_to_employee


class LostReceiptMCP:
    """The server may have saved it, but the reply never arrives."""

    async def call_tool(self, name, arguments):
        raise TimeoutError("no reply from the server")


class ReceiptMCP:
    def __init__(self, receipt, is_error=False):
        self.receipt = receipt
        self.is_error = is_error

    async def call_tool(self, name, arguments):
        return SimpleNamespace(structured_content=self.receipt, content=[], is_error=self.is_error)


def fallback(mcp):
    request = Request("E001", "anything")
    return asyncio.run(escalate(mcp, request, ReasonCode.AGENT_STEP_LIMIT, "no decision", print))


def test_lost_receipt_is_not_confirmed(capsys):
    assert fallback(LostReceiptMCP()).escalation_id is None


@pytest.mark.parametrize(
    "receipt",
    [
        pytest.param({"status": "open"}, id="no ID"),
        pytest.param({"escalation_id": "", "status": "open"}, id="empty ID"),
        pytest.param({"escalation_id": "ESC-1", "status": "closed"}, id="not open"),
    ],
)
def test_unexpected_receipt_is_not_confirmed(receipt, capsys):
    assert fallback(ReceiptMCP(receipt)).escalation_id is None


def test_thought_label_inside_a_block_is_removed(agent):
    llm = FakeLLM(
        reply(
            thought("Let me check.\n\nThought: Look up E001."),
            call("get_employee_info", {"employee_id": "E001"}),
        ),
        reply(check()),
        reply(submit()),
    )

    result = agent(llm)

    assert result.steps[0].thought == "Let me check. Look up E001."


def test_error_response_is_not_confirmed_even_with_an_id(capsys):
    receipt = {"escalation_id": "ESC-1", "status": "open"}

    assert fallback(ReceiptMCP(receipt, is_error=True)).escalation_id is None
