"""Tool functions tested directly: no MCP, no LLM (brief Part 3)."""

import sqlite3
from contextlib import closing
from datetime import UTC, datetime

import pytest

from equipment.tools import (
    check_request_eligibility,
    flag_for_human_review,
    get_employee_info,
    get_policy_limits,
)

# ---------------------------------------------------------------- get_employee_info


def test_employee_info_known_employee(repo, as_of):
    info = get_employee_info(repo, "E001", as_of)

    assert info == {
        "id": "E001",
        "name": "Alice Chen",
        "role": "employee",
        "tenure_months": 44,  # hired 2023-03-15
        "equipment": [  # oldest first
            {"item": "laptop", "issued_on": "2024-09-01", "age_months": 27},
            {"item": "monitor", "issued_on": "2025-06-01", "age_months": 18},
        ],
    }


def test_employee_info_lists_every_unit(repo, as_of):
    info = get_employee_info(repo, "E004", as_of)

    assert sorted(unit["item"] for unit in info["equipment"]) == [
        "laptop",
        "monitor",
        "monitor",
        "phone",
    ]
    assert all(unit["age_months"] == 12 for unit in info["equipment"])


def test_employee_info_unknown_id(repo, as_of):
    assert get_employee_info(repo, "E999", as_of)["error"] == "EMPLOYEE_NOT_FOUND"


def test_employee_info_ids_are_case_sensitive(repo, as_of):
    assert get_employee_info(repo, "e001", as_of)["error"] == "EMPLOYEE_NOT_FOUND"


@pytest.mark.parametrize("bad_id", ["", "   ", None])
def test_employee_info_empty_id(repo, as_of, bad_id):
    assert get_employee_info(repo, bad_id, as_of)["error"] == "INVALID_EMPLOYEE_ID"


def test_employee_info_strips_whitespace(repo, as_of):
    assert get_employee_info(repo, "  E002 ", as_of)["id"] == "E002"


def test_employee_info_no_equipment(repo, seeded_conn, as_of):
    seeded_conn.execute(
        "INSERT INTO employees VALUES ('E100', 'New Hire', 'employee', '2026-11-20')"
    )

    info = get_employee_info(repo, "E100", as_of)

    assert info["equipment"] == []
    assert info["tenure_months"] == 0


def test_employee_info_missing_issue_date(repo, seeded_conn, as_of):
    seeded_conn.execute(
        "INSERT INTO equipment (employee_id, item, issued_on) VALUES ('E005', 'headset', NULL)"
    )

    info = get_employee_info(repo, "E005", as_of)

    assert {"item": "headset", "issued_on": None, "age_months": None} in info["equipment"]


def test_employee_info_future_issue_date_has_no_age(repo, seeded_conn, as_of):
    seeded_conn.execute(
        "INSERT INTO equipment (employee_id, item, issued_on) "
        "VALUES ('E005', 'headset', '2027-01-15')"
    )

    info = get_employee_info(repo, "E005", as_of)

    assert {"item": "headset", "issued_on": "2027-01-15", "age_months": None} in info["equipment"]


# ---------------------------------------------------------------- get_policy_limits

ITEMS = ["headset", "keyboard_mouse", "laptop", "monitor", "phone"]
EXPECTED_LIMITS = {  # (max_qty, refresh_months) per item, from docs/requirements.md 2
    "intern": [(1, 24), (1, 24), (1, 36), (1, 24), (0, 0)],
    "employee": [(1, 24), (1, 24), (1, 36), (2, 24), (0, 0)],
    "manager": [(1, 24), (1, 24), (1, 24), (2, 24), (1, 24)],
    "contractor": [(1, 24), (1, 24), (0, 0), (1, 24), (0, 0)],
}


@pytest.mark.parametrize("role", EXPECTED_LIMITS)
def test_policy_limits_known_role(repo, role):
    result = get_policy_limits(repo, role)

    assert result["role"] == role
    assert [limit["item"] for limit in result["limits"]] == ITEMS
    limits = [(limit["max_qty"], limit["refresh_months"]) for limit in result["limits"]]
    assert limits == EXPECTED_LIMITS[role]


def test_policy_limits_new_role_needs_only_data(repo, seeded_conn):
    seeded_conn.execute("INSERT INTO policies VALUES ('executive', 'laptop', 2, 12)")

    result = get_policy_limits(repo, "executive")

    assert result["limits"] == [{"item": "laptop", "max_qty": 2, "refresh_months": 12}]


@pytest.mark.parametrize("role", ["ceo", "Manager", "", "   ", None])
def test_policy_limits_unknown_role(repo, role):
    assert get_policy_limits(repo, role)["error"] == "UNKNOWN_ROLE"


def test_policy_limits_strips_whitespace(repo):
    assert get_policy_limits(repo, "  manager ")["role"] == "manager"


# ---------------------------------------------------------------- check_request_eligibility


@pytest.mark.parametrize(
    ("employee_id", "item", "status", "reason_code"),
    [
        ("E001", "monitor", "eligible", "WITHIN_LIMIT"),
        ("E002", "laptop", "not_eligible", "ROLE_NOT_ELIGIBLE"),
        ("E003", "monitor", "not_eligible", "REFRESH_NOT_DUE"),
        ("E004", "laptop", "not_eligible", "REFRESH_NOT_DUE"),
        ("E006", "laptop", "eligible", "REFRESH_DUE"),
    ],
)
def test_eligibility_seed_employees(repo, as_of, employee_id, item, status, reason_code):
    result = check_request_eligibility(repo, employee_id, item, as_of)

    assert (result["status"], result["reason_code"]) == (status, reason_code)


def test_eligibility_full_result(repo, as_of):
    assert check_request_eligibility(repo, "E003", "monitor", as_of) == {
        "employee_id": "E003",
        "item": "monitor",
        "status": "not_eligible",
        "reason_code": "REFRESH_NOT_DUE",
        "basis": "refresh",
        "held": 2,
        "max_qty": 2,
        "oldest_age_months": 8,
        "refresh_months": 24,
        "missing": None,
    }


def test_eligibility_unknown_employee(repo, as_of):
    result = check_request_eligibility(repo, "E999", "laptop", as_of)

    assert result["error"] == "EMPLOYEE_NOT_FOUND"


@pytest.mark.parametrize("bad_id", ["", "   ", None])
def test_eligibility_empty_employee_id(repo, as_of, bad_id):
    result = check_request_eligibility(repo, bad_id, "laptop", as_of)

    assert result["error"] == "INVALID_EMPLOYEE_ID"


@pytest.mark.parametrize("item", ["standing_desk", "", None])
def test_eligibility_item_not_in_catalog(repo, as_of, item):
    result = check_request_eligibility(repo, "E001", item, as_of)

    assert (result["status"], result["reason_code"]) == ("needs_review", "ITEM_NOT_IN_CATALOG")


def test_eligibility_normalizes_input(repo, as_of):
    result = check_request_eligibility(repo, " E006 ", " Laptop ", as_of)

    assert (result["employee_id"], result["item"]) == ("E006", "laptop")


def test_eligibility_missing_policy_row(repo, seeded_conn, as_of):
    seeded_conn.execute("DELETE FROM policies WHERE role = 'employee' AND item = 'monitor'")

    result = check_request_eligibility(repo, "E001", "monitor", as_of)

    assert (result["reason_code"], result["missing"]) == ("RECORDS_INCOMPLETE", "role_policy")


def test_eligibility_employee_with_unknown_role(repo, seeded_conn, as_of):
    seeded_conn.execute("INSERT INTO employees VALUES ('E100', 'Pat Doe', 'ceo', '2020-01-01')")

    result = check_request_eligibility(repo, "E100", "laptop", as_of)

    assert (result["reason_code"], result["missing"]) == ("RECORDS_INCOMPLETE", "role_policy")


def test_eligibility_missing_issue_date(repo, seeded_conn, as_of):
    seeded_conn.execute("UPDATE equipment SET issued_on = NULL WHERE employee_id = 'E006'")

    result = check_request_eligibility(repo, "E006", "laptop", as_of)

    assert (result["reason_code"], result["missing"]) == ("RECORDS_INCOMPLETE", "issue_date")


def test_eligibility_new_item_needs_only_data(repo, seeded_conn, as_of):
    seeded_conn.execute("INSERT INTO policies VALUES ('employee', 'tablet', 1, 24)")

    result = check_request_eligibility(repo, "E001", "tablet", as_of)

    assert result["reason_code"] == "WITHIN_LIMIT"


# ---------------------------------------------------------------- flag_for_human_review

NOW = datetime(2026, 12, 1, 9, 30, tzinfo=UTC)


def flag(repo, **overrides):
    args = {
        "employee_id": "E003",
        "request": "One of my monitors died",
        "reason": "Breakage can't be verified",
        "reason_code": "EXCEPTION_REQUEST",
        "created_at": NOW,
    }
    return flag_for_human_review(repo, **(args | overrides))


def saved_escalations(db_path):
    """Read through a second connection, so uncommitted rows don't count."""
    with closing(sqlite3.connect(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM escalations")]


def test_flag_saves_the_escalation(repo, db_path):
    result = flag(repo)

    assert result["status"] == "open"
    assert saved_escalations(db_path) == [
        {
            "id": result["escalation_id"],
            "employee_id": "E003",
            "request": "One of my monitors died",
            "reason": "Breakage can't be verified",
            "reason_code": "EXCEPTION_REQUEST",
            "created_at": "2026-12-01T09:30:00+00:00",
            "status": "open",
        }
    ]


def test_flag_id_format(repo):
    prefix, suffix = flag(repo)["escalation_id"].split("-")

    assert prefix == "ESC"
    assert len(suffix) == 8


def test_flag_twice_saves_two(repo, db_path):
    first, second = flag(repo), flag(repo)

    assert first["escalation_id"] != second["escalation_id"]
    assert len(saved_escalations(db_path)) == 2


def test_flag_unknown_employee_is_saved(repo, db_path):
    flag(repo, employee_id="E999", reason_code="UNKNOWN_EMPLOYEE")

    assert saved_escalations(db_path)[0]["employee_id"] == "E999"


def test_flag_strips_whitespace(repo, db_path):
    flag(repo, employee_id=" E003 ", request=" broken ", reason=" unverifiable ")

    saved = saved_escalations(db_path)[0]
    assert (saved["employee_id"], saved["request"], saved["reason"]) == (
        "E003",
        "broken",
        "unverifiable",
    )


@pytest.mark.parametrize(
    "reason_code",
    [
        "UNKNOWN_EMPLOYEE",
        "VAGUE_REQUEST",
        "ITEM_NOT_IN_CATALOG",
        "RECORDS_INCOMPLETE",
        "CLAIM_CONTRADICTS_RECORDS",
        "EXCEPTION_REQUEST",
        "REFLECTION_FAILED",
        "AGENT_STEP_LIMIT",
        "AGENT_REFUSED",
    ],
)
def test_flag_accepts_escalation_codes(repo, reason_code):
    assert flag(repo, reason_code=reason_code)["status"] == "open"


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"employee_id": ""}, "INVALID_EMPLOYEE_ID"),
        ({"employee_id": "  "}, "INVALID_EMPLOYEE_ID"),
        ({"employee_id": None}, "INVALID_EMPLOYEE_ID"),
        ({"request": " "}, "INVALID_ESCALATION"),
        ({"request": None}, "INVALID_ESCALATION"),
        ({"reason": " "}, "INVALID_ESCALATION"),
        ({"reason": None}, "INVALID_ESCALATION"),
        ({"reason_code": None}, "INVALID_ESCALATION"),
        ({"reason_code": "MADE_UP"}, "INVALID_ESCALATION"),
        ({"reason_code": "WITHIN_LIMIT"}, "INVALID_ESCALATION"),
        ({"reason_code": "REFRESH_DUE"}, "INVALID_ESCALATION"),
        ({"reason_code": "REFRESH_NOT_DUE"}, "INVALID_ESCALATION"),
        ({"reason_code": "ROLE_NOT_ELIGIBLE"}, "INVALID_ESCALATION"),
        ({"reason_code": "AT_MAX_QUANTITY"}, "INVALID_ESCALATION"),
    ],
)
def test_flag_rejects_bad_input_and_saves_nothing(repo, db_path, overrides, error):
    result = flag(repo, **overrides)

    assert result["error"] == error
    assert saved_escalations(db_path) == []
