"""Tool functions tested directly: no MCP, no LLM (brief Part 3)."""

import pytest

from equipment.tools import get_employee_info

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
