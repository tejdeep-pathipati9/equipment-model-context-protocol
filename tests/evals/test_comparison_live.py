"""Ten natural requests, one run each. Explicitly select this file to limit live cost."""

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest
from evals.comparison import Comparison

from equipment.agent.llm import FoundryClient
from equipment.agent.mcp_client import connect
from equipment.agent.react import run_agent
from equipment.agent.types import Request

pytestmark = pytest.mark.eval
CASES = json.loads(Path("scenarios/comparison.json").read_text())


@pytest.fixture(scope="module")
def comparison_results(tmp_path_factory):
    directory = tmp_path_factory.mktemp("comparison-results")
    results = []
    yield directory, results
    print("\nCase | Before correct | After correct | Recorded | Calls before/total")
    for row in results:
        if "error" in row:
            print(f"{row['case']['id']} | ERROR: {row['error']}")
            continue
        print(
            f"{row['case']['id']} | {row['before_correct']} | {row['after_correct']} | "
            f"{row['recorded']} | {row['before_call_count']}/{len(row['calls'])}"
        )
    print(f"Completed {sum('error' not in row for row in results)}/{len(results)} attempted cases")
    print(
        f"Before correct: {sum(row.get('before_correct', False) for row in results)}/{len(results)}"
    )
    print(
        f"After correct: {sum(row.get('after_correct', False) for row in results)}/{len(results)}"
    )
    print(f"Results and manual message checks: {directory}")


@pytest.mark.parametrize("case", CASES, ids=[case["id"] for case in CASES])
def test_before_and_after_reflection(case, server_db, monkeypatch, comparison_results):
    monkeypatch.setenv("MCP_TRANSPORT", "stdio")
    comparison = Comparison(FoundryClient.from_env())
    request = Request(case["employee_id"], case["text"])
    directory, results = comparison_results

    async def go():
        async with connect() as mcp:
            return await run_agent(comparison, mcp, request, critic=comparison.review)

    try:
        result = asyncio.run(go())
    except Exception as error:
        # Preserve usage and the error without rerunning a costly case automatically.
        row = {
            "case": case,
            "error": type(error).__name__,
            "first_valid_draft": comparison.first_draft,
            "first_draft_evidence": comparison.first_steps,
            "calls": comparison.calls,
        }
        results.append(row)
        (directory / f"{case['id']}.json").write_text(json.dumps(row, indent=2))
        raise

    row = comparison.report(case, result)
    with closing(sqlite3.connect(server_db)) as conn:
        row["saved_escalations"] = conn.execute("SELECT COUNT(*) FROM escalations").fetchone()[0]
    results.append(row)
    (directory / f"{case['id']}.json").write_text(json.dumps(row, indent=2))

    assert row["after_correct"], row["final_decision"]
    assert result.recorded, "The final escalation was not confirmed as saved"
    assert row["saved_escalations"] == int(result.decision.decision == "escalate")
