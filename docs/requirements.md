# Requirements: IT Equipment Request Handler

An internal system that reads an employee's equipment request, checks it against their role's
policy and their equipment history, and **approves**, **denies**, or **escalates** it to a human
reviewer. Policy checks are done by deterministic code exposed as MCP tools; an LLM agent handles
the language (understanding the request, spotting exception claims and contradictions, drafting
the reply).

---

## 1. Request data

A request arrives as structured metadata plus free text:

```json
{ "employee_id": "E001", "text": "Can I get a second monitor? And can I have it by Friday?" }
```

| Field | Source | Notes |
|---|---|---|
| Employee | `employee_id` in **metadata** | Never parsed from the text. Identity claimed in free text is not trusted. |
| Role | Looked up from the employee record | One of `intern`, `employee`, `manager`, `contractor`. |
| Item requested | Extracted from the text by the agent | Must map to **exactly one** catalog item: `laptop`, `monitor`, `keyboard_mouse`, `headset`, `phone`. |
| Quantity | Extracted from the text by the agent | **Always one unit.** A request for several units or several items is not decided automatically (§4, rule 2). |
| Reason / claims | Extracted from the text by the agent | **Optional.** A plain request ("Can I get a headset?") is sufficient; no justification is needed for an in-policy request. Claims such as "it's broken" or "it's 4 years old" only matter for the exception and contradiction rules (§4). |

---

## 2. Policy rules

Each role has, per item, a **maximum quantity** held at once and a **refresh interval** (minimum
age before a unit may be replaced). Max quantity **0** means the role is not eligible for the item.

| Item | intern | employee | manager | contractor |
|---|---|---|---|---|
| laptop | 1 / 36 mo | 1 / 36 mo | 1 / 24 mo | **0** |
| monitor | 1 / 24 mo | 2 / 24 mo | 2 / 24 mo | 1 / 24 mo |
| keyboard_mouse | 1 / 24 mo | 1 / 24 mo | 1 / 24 mo | 1 / 24 mo |
| headset | 1 / 24 mo | 1 / 24 mo | 1 / 24 mo | 1 / 24 mo |
| phone | **0** | **0** | 1 / 24 mo | **0** |

Policies are stored as data (a database table), not hard-coded.

**A refresh is a replacement.** An approved refresh swaps the oldest unit for a new one; the old
unit is returned and the employee's holdings do not increase. A refresh never grants an additional
unit beyond the max.

### 2.1 Eligibility rules

Evaluated by code, against a fixed `as_of` date. The check to run is **inferred from the
employee's holdings**, not from the wording of the request: below the max, the question is "can
they have one more?"; at the max, the only meaningful question is "is it time to replace one?".
Rules are applied in order; the first match wins.

| # | Condition | Basis | Status | Reason code |
|---|---|---|---|---|
| 1 | item not in catalog | — | `needs_review` | `ITEM_NOT_IN_CATALOG` |
| 2 | role has no policy for this item (unknown role or missing policy row) | — | `needs_review` | `RECORDS_INCOMPLETE` |
| 3 | role's max quantity is 0 | — | `not_eligible` | `ROLE_NOT_ELIGIBLE` |
| 4 | units held < max | `quantity` | `eligible` | `WITHIN_LIMIT` |
| 5 | units held ≥ max, any unit of the item has a missing or future issue date | `refresh` | `needs_review` | `RECORDS_INCOMPLETE` |
| 6 | units held ≥ max, oldest unit's age ≥ refresh interval | `refresh` | `eligible` | `REFRESH_DUE` |
| 7 | units held ≥ max, oldest unit's age < refresh interval | `refresh` | `not_eligible` | `REFRESH_NOT_DUE` |

The result reports which check ran (`basis`) so the trace shows it, and for `RECORDS_INCOMPLETE`
what is missing (`role_policy` or `issue_date`). An unknown employee returns a structured
`EMPLOYEE_NOT_FOUND` error, and an empty ID returns `INVALID_EMPLOYEE_ID`; neither is an exception.

### 2.2 Dates and ages

- All date logic takes an explicit `as_of` date; business logic never reads the system clock.
  Tests and the demo use `as_of = 2026-12-01` so results are reproducible.
- **Age** (and tenure) is measured in **completed calendar months**. A month completes on the same
  day-of-month as the issue date; if that day doesn't exist in the month, it completes on the
  month's last day.
  - Issued 2024-12-01 → 24 months on 2026-12-01, 23 months on 2026-11-30.
  - Issued 2025-01-31 → 1 month on 2025-02-28.
- A refresh becomes due **on the refresh anniversary** (age ≥ refresh interval).
- An issue date after `as_of` is treated as unreliable data (rule 5 above).

---

## 3. MCP tools

| Tool | Input | Returns | Side effect |
|---|---|---|---|
| `get_employee_info` | `employee_id` | id, name, role, tenure in months, equipment on file (item, issue date, age in months; `null` if the date is missing) | none |
| `get_policy_limits` | `role` | each item's max quantity and refresh interval for that role; `UNKNOWN_ROLE` error otherwise | none |
| `check_request_eligibility` | `employee_id`, `item` | status, reason code, basis (`quantity` / `refresh`), details (held, max, oldest age, refresh interval, missing) | none |
| `flag_for_human_review` | `employee_id`, `request`, `reason`, `reason_code` | escalation ID, status `open`; `INVALID_ESCALATION` error if the request or reason is empty or the code isn't an escalation code | writes an escalation record |

**Notes on the brief:**
- `check_request_eligibility` keeps the brief's two-argument signature. Whether it checks
  quantity or refresh is inferred from holdings (§2.1), so the agent never has to classify the
  request type, and a replacement request from someone at their max is never misjudged as
  "one more".
- `flag_for_human_review` adds `reason_code` (a fixed code next to the free-text `reason`), so
  escalations can be counted and tested.
- Unknown employees can still be escalated; the escalation record does not require a valid ID.

---

## 4. Decision rules: approve, deny, escalate

The eligibility tool gives the **policy verdict**; the agent adds **language judgment**. Rules are
applied **in order**; the first match wins.

| # | Condition | Decision | Reason code |
|---|---|---|---|
| 1 | Employee not found | Escalate | `UNKNOWN_EMPLOYEE` |
| 2 | Request doesn't resolve to exactly one unit of one catalog item (too vague, several items, or several units) | Escalate | `VAGUE_REQUEST` |
| 3 | Eligibility status is `needs_review` | Escalate | eligibility reason code (`ITEM_NOT_IN_CATALOG`, `RECORDS_INCOMPLETE`) |
| 4 | A factual claim in the request contradicts the records (§4.1) | Escalate | `CLAIM_CONTRADICTS_RECORDS` |
| 5 | Approval would need an exception (`not_eligible`, **or** an additional unit at the max) **and** the employee claims breakage/loss/damage or asks for an exception | Escalate | `EXCEPTION_REQUEST` |
| 6 | Basis is `refresh` and the employee explicitly asks for an **additional** unit (keeping all current ones) | **Deny** | `AT_MAX_QUANTITY` |
| 7 | `not_eligible` | **Deny** | eligibility reason code |
| 8 | `eligible` | **Approve** | eligibility reason code |

### 4.1 Contradictions

A claim contradicts the records when:
- **Age:** the stated age (converted to months; "4 years" = 48, "about 2 years" = 24) differs by
  **more than 12 months** from the recorded age of **every** unit of that item. With several
  units, matching any one unit is enough. Approximate words ("about", "almost") are covered by the
  12-month tolerance.
- **Holdings:** the employee claims to have, or not have, an item and the records disagree (e.g.
  "my phone broke" with no phone on file; "I don't have a monitor" with one on file).

Subjective descriptions ("old", "slow", "ancient") are not factual claims and never contradict.

### 4.2 Broken equipment

A breakage, loss or damage claim **only matters when approval depends on it**. If normal policy
already permits the request (e.g. "my monitor broke" from someone below their max, or due for a
refresh), it is **approved under normal policy** without verifying the claim. Human review is
needed only when the claim is the basis for an exception (rule 5).

**New or replacement below the max.** If the employee is below their max, the request is approved
as one more unit even if the wording could mean a replacement: policy allows it either way, so the
decision is not ambiguous. Example: max 2, holds one 29-month-old monitor, "I want a new monitor".

### 4.3 What counts as ambiguous

A request is **escalated rather than decided** when the system cannot decide it reliably from its
own data:
- the requester, the item or the quantity cannot be identified (rules 1–2);
- the records needed to decide are missing or unreliable (rule 3);
- the request's facts conflict with the records (rule 4);
- approval depends on a claim the system cannot verify, such as breakage or loss (rule 5).

**Deny vs. escalate.** Outside policy does not always mean escalate. A request is **denied** when it
is clearly outside policy and the data is unambiguous (e.g. a contractor asking for a laptop). It
is **escalated** when it is outside policy *and* there is a reason a human might decide
differently (an unverifiable breakage claim, a contradiction).

Every escalation is saved with `flag_for_human_review`, and the reply includes its escalation ID.

---

## 5. Response

```json
{
  "decision": "approve | deny | escalate",
  "reason_code": "WITHIN_LIMIT",
  "item": "monitor",
  "message_to_employee": "…",
  "escalation_id": null
}
```

The message to the employee must:
- state only facts returned by the tools;
- promise nothing the tools did not confirm: **no delivery dates, no costs, no outcome of a human
  review**;
- for basis `refresh`, say it replaces an existing unit; for basis `quantity`, not say that a unit
  is replaced or must be returned;
- when escalated, explain why a person needs to decide. The system adds "Sent for human review"
  and the escalation ID once the escalation is saved; the agent never writes them.

**The agent decides; the system writes, once.** The agent only looks up facts and submits a draft.
To escalate, the draft has a `reason_code` and an `escalation_reason` (a note for the human
reviewer), but no escalation ID. Nothing is saved until the draft passes reflection:
- a code check confirms the decision matches the eligibility result **for the draft's item**
  (`approve` ⇒ `eligible`; `deny` ⇒ `not_eligible`, or `AT_MAX_QUANTITY` with basis `refresh`;
  `escalate` ⇒ the reason code fits the results in rule order, e.g. `EXCEPTION_REQUEST` is rejected
  when policy allows one more unit);
- a separate LLM critic checks the decision, the message and the reviewer's note against the tool
  observations.

Only then does the system call `flag_for_human_review`, once, and add the reference to the message.
If the save fails, the reply says it could not be recorded. A draft that fails review goes back to
the agent with its issues, so it can reword it or look up a missing fact; nothing was saved, so it
may still change its decision. After 2 rejections, the next one escalates the request instead
(`REFLECTION_FAILED`). If the critic gives no verdict on an escalation that passed the code check,
it is saved with its proposed reason code, a standard message, and a reviewer's note saying it was
not reviewed. If the agent exceeds 8 reasoning steps, the request is escalated
(`AGENT_STEP_LIMIT`). If the model declines to answer, it is escalated (`AGENT_REFUSED`).

---

## 6. Test data and acceptance scenarios

Synthetic employees (equipment as of 2026-12-01):

| ID | Role | Equipment on file |
|---|---|---|
| E001 | employee | 1 monitor (2025-06-01), laptop (2024-09-01) |
| E002 | contractor | monitor (2026-02-01) |
| E003 | employee | 2 monitors (2026-03-15), laptop (2025-01-10) |
| E004 | manager | laptop, 2 monitors, phone (all 2025-12-01) |
| E005 | intern | laptop (2026-06-01) |
| E006 | employee | laptop (2023-05-01) |

The system is accepted when these requests produce these outcomes:

| # | Employee | Request | Decision | Reason code |
|---|---|---|---|---|
| 1 | E001 | "Can I get a second monitor? And can I have it by Friday?" | Approve | `WITHIN_LIMIT` (reply must not promise Friday) |
| 2 | E002 | "I need a company laptop for my project." | Deny | `ROLE_NOT_ELIGIBLE` |
| 3 | E003 | "One of my monitors died, can I get a replacement?" | Escalate | `EXCEPTION_REQUEST` |
| 4 | E004 | "My laptop is 4 years old and really slow, I need a new one." | Escalate | `CLAIM_CONTRADICTS_RECORDS` |

Additional tool-level cases (unit tests):
- unknown employee ID; unknown item → `ITEM_NOT_IN_CATALOG`; unknown role → `UNKNOWN_ROLE`;
- contractor laptop → `ROLE_NOT_ELIGIBLE`; E001 monitor (1 of 2) → `WITHIN_LIMIT`;
- E006 laptop (1 of 1, 43 months) → `REFRESH_DUE`; E003 monitor (2 of 2, 8 months) →
  `REFRESH_NOT_DUE`;
- refresh exactly on the anniversary → `REFRESH_DUE`; one day before → `REFRESH_NOT_DUE`;
- month-end issue date (2025-01-31 → 1 month on 2025-02-28);
- missing policy row, missing issue date, future issue date → `RECORDS_INCOMPLETE`.
