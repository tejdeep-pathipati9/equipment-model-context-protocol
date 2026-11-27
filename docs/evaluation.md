# Paired reflection evaluation

The set in `scenarios/comparison.json` contains ten natural requests, using the seeded records
as of 2026-12-01. Five should be decided automatically (three approvals and two denials), and five
should be escalated. It includes all four required demo scenarios. Run each case once initially.

Each case uses the normal ReAct loop and real MCP tools on its own temporary database. Capture
the first draft that passes schema validation and deterministic checks, then continue the same
conversation through the live critic and bounded revisions. Both stages receive the message
rules from the start. The baseline draft is captured before any critic feedback and before any
escalation is saved. There is one investigation and one final commit per case.

This measures the change from ReAct with code checks to the same run after LLM reflection. It
does not compare two independently generated runs. The baseline has no escalation receipt because
it is a draft. Compare its decision, reason code, message and reviewer note with the final answer;
check persistence separately on the final result.

## Run and results

With Foundry settings already exported, explicitly select the comparison file to run only ten
cases. The following command makes paid model calls. It does not load an environment file.

```sh
uv run pytest -m eval -s tests/evals/test_comparison_live.py
```

For an initial cost measurement, select just the first case.

```sh
uv run pytest -m eval -s tests/evals/test_comparison_live.py -k 01-second-monitor
```

The terminal prints a before/after outcome table and the temporary results directory. Each JSON
file contains the labeled request, both answers, evidence, reviews, full tool trace, usage for
each model call, latency and saved-ticket count. Copy these files from the printed directory into
your submission evidence before cleaning temporary files. Exact decision and reason-code checks
run automatically, with item checks where the item is unambiguous. A wrong final answer or failed
handoff fails the test after the evidence is written. A run that never reaches a code-valid draft
has a missing baseline and counts as unsuccessful. Exceptions are recorded and tests are not
automatically repeated by this runner.

Ordinary `pytest` excludes live evaluations. Selecting all tests with `-m eval` also runs the
existing planted-draft suite, which costs additional calls. The manual CI evaluation job uses
that broader selection. The comparison command above selects only this set.

## Grading and interpretation

For each case, manually grade both answers against its `message_checks`, tool evidence and the
message rules in `docs/requirements.md`. Check the reviewer note too. Fill in
`manual_message_review` with a pass/fail for each answer and notes quoting unsupported statements.
A valid system-added handoff receipt in the final message is allowed. A claim in the baseline
that the escalation was already sent is unsupported. Use the same rubric for both answers.

Report decision accuracy, reason-code accuracy, missed and unnecessary escalations, message
pass rates, corrections, regressions and unchanged answers. The automatic `decision_change`
field concerns labeled outcome correctness only; a wording correction needs manual grading.
Count step limits, refusals and outages as unsuccessful expected outcomes even if their fallback
successfully saves a ticket. Include every attempted case in the denominator.

Compare `before_usage` with `total_usage` and `before_seconds` with `total_seconds`. Cache creation
and cache reads are separate from ordinary input tokens. SDK retries can add requests and cost
that this wrapper cannot observe individually; a failed call may have unknown billed usage.
The wrapper records SDK calls, not wire attempts. Apply your actual model rates to usage if you
need dollar costs. Manual grading adds no model calls.

This is a small, deliberately balanced development set. These cases overlap with existing demos
and tests, so report them as an initial evaluation rather than an unseen benchmark. One run per
case does not establish repeatability or a production traffic success rate. Record the git
revision and deployment/model version with the results. Keep the existing planted-draft suite
as separate evidence that the critic catches specific known mistakes.

## One planted decision error

The manager-exception request also has a separate planted-draft recovery test. Its hand-written
draft correctly cites the laptop's age and refresh interval but denies the request, overlooking
the exception claim. The code check passes because the denial matches the eligibility result.
The critic must identify the missed exception, then the real drafter must change the decision to
`escalate / EXCEPTION_REQUEST`. The test checks that exactly one ticket is saved.

With settings already exported, run only this diagnostic with the following paid command.

```sh
uv run pytest -m eval -s tests/evals/test_reflection_live.py -k claimed_exception_revises_policy_only_denial
```

The first lookup and denial are scripted. The critic and subsequent drafter turns use the live
model. The test records the planted draft, all reviews, final result, trace and actual model usage
in its printed temporary directory. Label this as recovery from an injected error in the report.
The same request was answered correctly in the natural ten-case run. Keep those original results
and this diagnostic separate when reporting accuracy and cost.
