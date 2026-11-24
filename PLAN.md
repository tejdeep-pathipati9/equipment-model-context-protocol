# plan.md — IT Equipment Request Handler (MCP Server + ReAct Agent)

## 1. Goal

Build an internal IT equipment request handler from scratch:

- An **MCP server** that exposes employee, policy, eligibility and escalation tools over a mock SQLite database.
- A **ReAct agent** (Claude Opus via Azure Foundry) that reads a natural-language request, investigates with those tools, and **approves, denies, or escalates** it, with a reflection step before finalizing.
- Packaged in **Docker** and verified through **GitHub Actions**.

**Guiding principle:** deterministic policy logic lives in plain, unit-tested Python. The LLM handles only language: mapping free text to catalog items, spotting exception claims and contradictions, and drafting the reply.

---

## 2. Scope

| Tier | Contents |
|---|---|
| **Tier 1: must ship** | Requirements + policy rules · minimal one-tool server/client · 4 tools (plain functions → tests → MCP wrappers) · ReAct agent with printed trace · reflection (LLM critic + code check) · 4 demo scenarios · Dockerfile · GitHub Actions `test` job green |
| **Tier 2: cheap quality** | `agent-eval` CI job (live LLM, outcome-based) · hand-written bad-draft critic test |
| **Tier 3: stretch** | More escalation types (multi-item, accessibility) · HTTP transport / split containers · OIDC auth instead of an API key secret |

**Rule:** no Tier 3 work until every Tier 1 deliverable (screenshot) exists.
**If behind, cut in this order:** `agent-eval` CI job (run manually and screenshot instead) → revisions from 2 to 1.

---

## 3. Requirements

> **`docs/requirements.md` is the source of truth for the rules.** It supersedes this section where they differ: one unit per request, `RECORDS_INCOMPLETE` (missing policy / issue date), completed-calendar-month ages, contradiction tolerance (±12 months), and `AT_MAX_QUANTITY` as a decision code (deny an additional unit at max).

### 3.1 Request input

A request arrives as structured metadata plus free text:

```json
{ "employee_id": "E001", "text": "Can I get a second monitor?" }
```

- `employee_id` comes from **metadata**, never parsed from the text. Identity claimed in free text is not trusted.
- The agent maps the text to **one catalog item**. It does **not** classify a request type: the
  eligibility check infers it from holdings (§3.5).

### 3.2 Roles and catalog

- **Roles:** `intern`, `employee`, `manager`, `contractor`
- **Catalog:** `laptop`, `monitor`, `keyboard_mouse`, `headset`, `phone`

### 3.3 Policy rules (item × role → max quantity + refresh interval)

| Item | intern | employee | manager | contractor |
|---|---|---|---|---|
| laptop | 1 / 36 mo | 1 / 36 mo | 1 / 24 mo | **0** (not eligible) |
| monitor | 1 / 24 mo | 2 / 24 mo | 2 / 24 mo | 1 / 24 mo |
| keyboard_mouse | 1 / 24 mo | 1 / 24 mo | 1 / 24 mo | 1 / 24 mo |
| headset | 1 / 24 mo | 1 / 24 mo | 1 / 24 mo | 1 / 24 mo |
| phone | **0** | **0** | 1 / 24 mo | **0** |

Max quantity 0 = role not eligible for the item. Policies are **data** (a DB table), not code.

### 3.4 Dates

All date logic takes an explicit `as_of: date`. Nothing calls `datetime.now()` inside business logic.
- Tests pass fixed dates.
- The demo and CI use `AS_OF_DATE=2026-12-01`, so traces are reproducible.
- Production would pass today's date from the composition root.

### 3.5 Eligibility result (from `check_request_eligibility`)

```json
{
  "status": "eligible | not_eligible | needs_review",
  "reason_code": "WITHIN_LIMIT",
  "basis": "quantity | refresh | null",
  "employee_id": "E001", "item": "monitor",
  "held": 1, "max_qty": 2, "oldest_age_months": null, "refresh_months": 24,
  "missing": null
}
```

| Status | Reason codes |
|---|---|
| `eligible` | `WITHIN_LIMIT` (held < max, basis `quantity`) · `REFRESH_DUE` (held ≥ max, oldest age ≥ interval, basis `refresh`) |
| `not_eligible` | `ROLE_NOT_ELIGIBLE` (max = 0) · `REFRESH_NOT_DUE` (held ≥ max, oldest age < interval, basis `refresh`) |
| `needs_review` | `ITEM_NOT_IN_CATALOG` |
| error | `EMPLOYEE_NOT_FOUND` (returned as a structured error, not an exception across MCP) |

### 3.6 Decision rules: approve / deny / escalate

The eligibility tool gives the **policy verdict**. The agent adds **language judgment**. Rules are applied in this order:

| # | Condition | Decision | Escalation reason |
|---|---|---|---|
| 1 | Employee not found | Escalate | `UNKNOWN_EMPLOYEE` |
| 2 | Request too vague to map to one catalog item | Escalate | `VAGUE_REQUEST` |
| 3 | Status `needs_review` | Escalate | `ITEM_NOT_IN_CATALOG` |
| 4 | User's factual claim contradicts records (e.g., "4 years old" vs. issued 12 months ago) | Escalate | `CLAIM_CONTRADICTS_RECORDS` |
| 5 | `not_eligible` **and** user claims broken/lost/damaged or asks for an exception | Escalate | `EXCEPTION_REQUEST` |
| 6 | `not_eligible`, no exception claim | **Deny** | — |
| 7 | `eligible`, no contradiction | **Approve** | — |

**Deny vs. escalate:** deny when the request is clearly outside policy and the data is unambiguous. Escalate when facts conflict, the system cannot verify a claim (like breakage), or a human judgment call is needed.

**Every escalation** calls `flag_for_human_review` and returns its escalation ID.

### 3.7 Final output

```json
{
  "decision": "approve | deny | escalate",
  "reason_code": "WITHIN_LIMIT",
  "item": "monitor",
  "message_to_employee": "…",
  "escalation_id": null
}
```

The message must not promise anything the tools didn't confirm: no delivery dates, no costs, no outcome of a human review.

---

## 4. Architecture

### 4.1 Layers

```
 request (employee_id + text)
        │
        ▼
 ┌──────────────┐  Anthropic SDK   ┌───────────────────────┐
 │  Agent (CLI) │ ───────────────▶ │ Claude Opus (Foundry) │
 │  ReAct loop  │ ◀─────────────── └───────────────────────┘
 │  Reflection  │
 └──────┬───────┘
        │ MCP over stdio (agent spawns server)
        ▼
 ┌──────────────┐
 │  MCP server  │  thin FastMCP wrappers, no logic
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │ Tool service │  4 plain functions (unit-tested)
 └──────┬───────┘
        ▼
 ┌──────────────┐     ┌──────────────┐
 │ Policy logic │     │  Repository  │──▶ SQLite
 │ (pure funcs) │     │  (Protocol)  │
 └──────────────┘     └──────────────┘
```

### 4.2 Project layout

```
equipment-agent/
├── pyproject.toml            # uv; scripts: seed, server, agent
├── uv.lock
├── Dockerfile
├── .dockerignore
├── .env.example
├── .github/workflows/ci.yml
├── docs/requirements.md      # section 3, standalone, for the deliverable
├── src/
│   ├── domain/
│   │   ├── models.py         # dataclasses + enums (Role, Item, Status, ReasonCode, Decision)
│   │   └── policy.py         # pure eligibility logic: evaluate(...) -> EligibilityResult
│   ├── data/
│   │   ├── repository.py     # Repository Protocol
│   │   ├── sqlite_repo.py    # SQLite implementation
│   │   ├── schema.sql
│   │   └── seed.py
│   ├── tools.py              # the 4 plain tool functions (repo + as_of injected)
│   ├── server.py             # FastMCP wrappers + composition root for the server
│   └── agent/
│       ├── llm.py            # LLMClient Protocol + FoundryClient
│       ├── mcp_client.py     # stdio connect, list tools, call tool, schema conversion
│       ├── react.py          # the ReAct loop
│       ├── prompts.py        # system prompt (decision rules), critic prompt
│       ├── reflection.py     # LLM critic + deterministic consistency check
│       ├── trace.py          # Thought / Action / Observation printer
│       └── cli.py            # entrypoint + composition root for the agent
├── scenarios/demo.json       # the 4 demo requests + expected outcomes
└── tests/
    ├── test_policy.py
    ├── test_tools.py         # each tool: normal + edge cases
    ├── test_reflection.py    # consistency check + critic with fake LLM / bad draft
    ├── test_mcp_smoke.py     # real MCP client → server, no LLM
    └── evals/test_agent_scenarios.py   # live LLM (agent-eval job only)
```

### 4.3 SOLID, applied pragmatically

| Principle | Where it shows up | What we deliberately **don't** do |
|---|---|---|
| **S**: single responsibility | Policy math (`policy.py`), storage (`sqlite_repo.py`), protocol adapter (`server.py`), loop (`react.py`), and review (`reflection.py`) each have one reason to change | No "manager" or "service" classes that do everything |
| **O**: open/closed | Roles, items and limits are DB rows; new reflection checks are added to a list of check functions; new escalation reasons are new enum values + a prompt line | No plugin system or rule engine |
| **L**: Liskov | Any `Repository` or `LLMClient` (real or test fake) honors the same contract: same return types, same "not found" behavior | No inheritance hierarchies to get wrong |
| **I**: interface segregation | Small Protocols: `Repository` (~5 methods), `LLMClient` (1 method) | No god-interface; no Protocol for things with only one implementation and no test need |
| **D**: dependency inversion | Tools take a `Repository` and `as_of`; the agent takes an `LLMClient` and an MCP session. Wiring happens only in two composition roots: `server.py` main and `cli.py` | No DI framework or factories; plain constructor/function arguments |

**Abstraction budget:** exactly **three seams**: `Repository`, `LLMClient`, and the `as_of` date. Everything else is concrete functions and dataclasses.

---

## 5. Data model (SQLite)

```sql
employees  (id TEXT PK, name TEXT, role TEXT, hire_date DATE)
equipment  (id INTEGER PK, employee_id TEXT FK, item TEXT, issued_on DATE)
policies   (role TEXT, item TEXT, max_qty INTEGER, refresh_months INTEGER, PK(role, item))
escalations(id TEXT PK, employee_id TEXT, request TEXT, reason TEXT,
            reason_code TEXT, created_at TIMESTAMP, status TEXT DEFAULT 'open')
```

- `uv run seed` creates the schema and loads seed data **if the DB file doesn't exist**.
- DB path comes from `EQUIP_DB_PATH` (default `./data/equipment.db`).
- Tests create a fresh temp DB per test via a pytest fixture.
- `escalations.employee_id` is not a foreign key, so an unknown employee can still be escalated.

### Seed data (as of 2026-12-01)

| ID | Role | Equipment on file | Used for |
|---|---|---|---|
| E001 | employee | 1 monitor (2025-06-01), laptop (2024-09-01) | Scenario 1: approve |
| E002 | contractor | monitor (2026-02-01) | Scenario 2: deny |
| E003 | employee | 2 monitors (2026-03-15), laptop (2025-01-10) | Scenario 3: escalate (broken) |
| E004 | manager | laptop (2025-12-01), 2 monitors (2025-12-01), phone (2025-12-01) | Scenario 4: escalate (contradiction) |
| E005 | intern | laptop (2026-06-01) | Edge-case tests |
| E006 | employee | laptop (2023-05-01) | Edge-case test: refresh due |

---

## 6. MCP server: tools

| Tool | Input | Returns | Side effect |
|---|---|---|---|
| `get_employee_info` | `employee_id` | `{id, name, role, tenure_months, equipment: [{item, issued_on, age_months}]}` or `EMPLOYEE_NOT_FOUND` error | none |
| `get_policy_limits` | `role` | `{role, limits: [{item, max_qty, refresh_months}]}` or `UNKNOWN_ROLE` error | none |
| `check_request_eligibility` | `employee_id`, `item` | Eligibility result (§3.5) | none |
| `flag_for_human_review` | `employee_id`, `request`, `reason`, `reason_code` | `{escalation_id, status: "open"}` | inserts a row in `escalations` |

Notes:
- `check_request_eligibility` keeps the brief's two-argument signature. Quantity vs. refresh is inferred from holdings (held < max → quantity check; held ≥ max → refresh check), and the result reports `basis`. A `request_type` parameter defaulting to `new` was rejected: it silently misjudges everyone at their max (e.g. E006's due laptop refresh → `AT_MAX_QUANTITY`).
- Wrappers in `server.py` only convert arguments and results. All logic is in `tools.py`.
- Transport: **stdio**. The agent starts the server as a subprocess.

---

## 7. Agent design

### 7.1 MCP client
- Spawns `python -m equipment.server` over stdio, initializes the session, and calls `list_tools()`.
- Converts MCP tool schemas into Anthropic `tools` format, so the model sees exactly what the server exposes.

### 7.2 ReAct loop (native tool use)
1. The system prompt contains the decision rules (3.6) and requires a short `Thought:` text block before every tool call.
2. Loop: send messages → for each `tool_use` block, print **Thought** (the preceding text), **Action** (tool + args), call the MCP tool, print **Observation** (result) → append `tool_result` → repeat.
3. Stop when the model returns a final answer with no tool calls (the draft, as JSON per §3.7), or when **max 8 steps** is reached. Hitting the limit forces an escalation with `AGENT_STEP_LIMIT`.
4. Model from `AGENT_MODEL` (Foundry deployment name); temperature 0.

### 7.3 Reflection (before finalizing)
1. **Deterministic consistency check (code, runs first):**
   - `approve` ⇒ last eligibility status was `eligible`
   - `deny` ⇒ last eligibility status was `not_eligible`, or reason `AT_MAX_QUANTITY` with basis `refresh`
   - `escalate` ⇒ `flag_for_human_review` was called and `escalation_id` is present
   - `reason_code` is a known enum value
2. **LLM critic (separate call):** receives the draft plus the raw tool observations and answers a fixed checklist:
   - Is every factual claim backed by an observation?
   - Does the decision match the eligibility verdict and the decision rules?
   - Does the message promise anything unconfirmed (dates, delivery, costs, review outcome)?
   - Returns `{verdict: "pass" | "revise", issues: [...]}`
3. On `revise` → the drafter revises with the issues → re-check. **Max 2 revisions.**
4. Still failing after 2 → final decision becomes **escalate** with `REFLECTION_FAILED`. The fallback is always the safe path.
5. Every check result and critic verdict is printed in the trace (this is the reflection evidence).

### 7.4 Making reflection evidence reliable
- **Live bait:** scenario 1's text adds "…and can I have it by Friday?" No tool knows delivery dates, so a draft promising Friday should be flagged.
- **Guaranteed test:** `test_reflection.py` feeds the critic a hand-written bad draft ("Approved!" when the tool said `not_eligible`) and asserts that the code check fails and the result is `revise`.
- The rubric accepts "improves **or confirms**", so a `pass` verdict also counts.
- **Make the live catch likely:** at temperature 0 a drafter told "never promise dates" will write a clean draft and the trace only shows `pass`. Put the no-promises / facts-only rules in the **critic** prompt, not the drafter's, so scenario 1's first draft is likely to mention Friday and get revised.
- **Screenshot-ready trace:** print the code-check result, the critic verdict + issues, and the draft **before and after** revision, so "improves" is visible on one screen.

---

## 8. Demo scenarios

| # | Employee | Request text | Expected | Reason |
|---|---|---|---|---|
| 1 | E001 (employee, 1 monitor) | "Can I get a second monitor? And can I have it by Friday?" | **Approve** | `WITHIN_LIMIT`; reflection removes the Friday promise |
| 2 | E002 (contractor) | "I need a company laptop for my project." | **Deny** | `ROLE_NOT_ELIGIBLE` |
| 3 | E003 (employee, 2 monitors, 8 months old) | "One of my monitors died, can I get a replacement?" | **Escalate** | `EXCEPTION_REQUEST` (refresh not due, breakage unverifiable) |
| 4 | E004 (manager, laptop 12 months old) | "My laptop is 4 years old and really slow, I need a new one." | **Escalate** | `CLAIM_CONTRADICTS_RECORDS` |

Stored in `scenarios/demo.json` with expected decision + reason code, used by both the demo runner and `agent-eval`.

---

## 9. Testing strategy

### 9.1 Unit tests (plain functions, no MCP, no LLM)

| Target | Normal cases | Edge cases |
|---|---|---|
| `policy.evaluate` | within limit; refresh due | max 0; held = 0; refresh exactly on boundary month; refresh not due; held > max (still refresh basis) |
| `get_employee_info` | known employee with equipment | unknown ID; employee with no equipment; empty/whitespace ID |
| `get_policy_limits` | each role | unknown role; case sensitivity |
| `check_request_eligibility` | E001 monitor → `WITHIN_LIMIT`; E006 laptop → `REFRESH_DUE` | unknown employee; unknown item → `needs_review`; contractor laptop → `ROLE_NOT_ELIGIBLE`; E003 monitor → `REFRESH_NOT_DUE` |
| `flag_for_human_review` | row inserted, ID returned | unknown employee still recorded; empty reason rejected; two calls → two distinct IDs |
| Consistency check | valid approve/deny/escalate | approve with `not_eligible`; escalate without ID; unknown reason code |
| Critic (fake `LLMClient`) | pass path parsed | bad draft → `revise`; malformed critic output → treated as `revise` |

### 9.2 MCP smoke test
`test_mcp_smoke.py` starts the real server over stdio with a temp DB, asserts that all 4 tools are listed, and calls each once. No LLM.

### 9.3 Agent evals (live LLM)
`tests/evals/` runs the 4 scenarios against Foundry and asserts **decision + reason_code only**, never the message wording. Marked `@pytest.mark.eval` and excluded from the default run.

---

## 10. Docker

- **One image** containing agent + server (stdio requires it).
- `python:3.12-slim` + uv binary; `uv sync --frozen --no-dev`.
- Non-root user.
- Secrets **only at runtime**: `docker run --env-file .env …`. `.env` is listed in `.dockerignore` and never copied.
- DB at `/app/data`, seeded on container start if missing; mount `-v ./data:/app/data` so escalations persist.
- Entrypoint = agent CLI:
  - `docker run --env-file .env -v ./data:/app/data equip-agent --scenario 3`
  - `docker run --env-file .env -v ./data:/app/data equip-agent --employee E004 --text "…"`

---

## 11. CI: GitHub Actions

| Job | Trigger | Steps | Secrets |
|---|---|---|---|
| `test` (required) | every push + PR | `uv sync` → lint (ruff) → `pytest -m "not eval"` (unit + smoke) → `docker build` | none |
| `agent-eval` | push to `main` + `workflow_dispatch` | build image → run `pytest -m eval` inside it | `ANTHROPIC_FOUNDRY_API_KEY`, `ANTHROPIC_FOUNDRY_RESOURCE`, `AGENT_MODEL` |

`agent-eval` uses outcome assertions and one retry on transient API errors. It is kept separate so that rate limits or cost never block PRs.

---

## 12. Configuration

| Variable | Purpose | Default |
|---|---|---|
| `ANTHROPIC_FOUNDRY_API_KEY` | Foundry auth | — (required for agent) |
| `ANTHROPIC_FOUNDRY_RESOURCE` | Foundry resource name | — (required for agent) |
| `AGENT_MODEL` | Foundry **deployment name** for Opus | — (required for agent) |
| `EQUIP_DB_PATH` | SQLite path | `./data/equipment.db` |
| `AS_OF_DATE` | Fixed "today" for policy logic | today; demo/CI use `2026-12-01` |
| `AGENT_MAX_STEPS` | ReAct step limit | `8` |
| `REFLECTION_MAX_REVISIONS` | Critic revision limit | `2` |

---

## 13. Build order

Each step must be confirmed working before the next one starts. If something breaks, go back to the last confirmed step.

| # | Step | Done when | Deliverable |
|---|---|---|---|
| 1 | Requirements doc (`docs/requirements.md`) | Rules in §3 written and reviewed | 📸 requirements + policy rules |
| 2 | Repo scaffold: uv, ruff, pytest, `test` CI job | Empty test suite passes in CI | — |
| 3 | Minimal server with one trivial tool + client script | Client lists the tool and gets a response | 📸 one-tool connection |
| 4 | SQLite schema, seed, repository | Seed creates DB; repo tests pass | — |
| 5 | `get_employee_info` as plain function → tests → MCP wrapper → call via client | Real response through MCP | — |
| 6 | Remaining 3 tools, same pattern, one at a time | All unit tests + smoke test pass | 📸 server code, tests, passing output |
| 7 | Dockerfile | `docker run … --help` works; CI builds image | — |
| 8 | Foundry client + one hardcoded tool call | Model calls a tool and gets a result | — |
| 9 | ReAct loop + trace printer | All 4 scenarios give expected decisions | 📸 agent code, 4 traces, 4 decisions |
| 10 | Reflection: code check → critic → revision loop | Bad-draft test passes; reflection appears in traces | 📸 reflection evidence |
| 11 | `agent-eval` CI job | Green run on `main` | 📸 passing pipeline |

---

## 14. Open questions and assumptions
- **Scenarios (assumed):** the 4 scenarios in 8 as proposed.
- **DB seeding (assumed):** seed on container start if missing, not baked into the image, so the demo DB can be reset by deleting the file.
- **One item per request (assumed):** multi-item requests are Tier 3. Until then, the agent escalates them as `VAGUE_REQUEST`.
