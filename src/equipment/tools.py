"""The plain tool functions. The repository and `as_of` date are passed in; no MCP, no globals.

Errors are returned as data ({"error": CODE, "message": ...}), never raised, so they reach the
agent as an observation it can reason about.

get_employee_info and get_policy_limits are implemented; check_request_eligibility and
flag_for_human_review follow.
"""

from dataclasses import asdict
from datetime import date
from typing import Any

from equipment.data.sqlite_repo import SqliteRepository
from equipment.domain.models import ErrorCode
from equipment.domain.policy import completed_months


def _error(code: ErrorCode, message: str) -> dict[str, Any]:
    return {"error": code, "message": message}


def _months_since(start: date | None, as_of: date) -> int | None:
    """Completed months, or None when the date is missing or in the future (unreliable)."""
    if start is None or start > as_of:
        return None
    return completed_months(start, as_of)


def get_employee_info(repo: SqliteRepository, employee_id: str, as_of: date) -> dict[str, Any]:
    """Role, tenure and current equipment on file for an employee.

    `as_of` is "today" for tenure and ages, passed in (not the clock) so results are repeatable.
    Ages let the agent catch false claims: "my laptop is 4 years old" vs. age_months=12.
    """
    employee_id = (employee_id or "").strip()
    if not employee_id:
        return _error(ErrorCode.INVALID_EMPLOYEE_ID, "employee_id must not be empty")

    employee = repo.get_employee(employee_id)
    if employee is None:
        return _error(ErrorCode.EMPLOYEE_NOT_FOUND, f"no employee with id {employee_id!r}")

    return {
        "id": employee.id,
        "name": employee.name,
        "role": employee.role,
        "tenure_months": _months_since(employee.hire_date, as_of),
        "equipment": [
            {
                "item": unit.item,
                "issued_on": unit.issued_on.isoformat() if unit.issued_on else None,
                "age_months": _months_since(unit.issued_on, as_of),
            }
            for unit in repo.list_equipment(employee.id)
        ],
    }


def get_policy_limits(repo: SqliteRepository, role: str) -> dict[str, Any]:
    """Max quantity and refresh interval per item for a role, e.g. contractor laptop: max_qty=0.

    A role is known if it has policy rows, so adding one needs no code change.
    """
    role = (role or "").strip()
    limits = repo.list_policies(role)
    if not limits:
        return _error(ErrorCode.UNKNOWN_ROLE, f"no policy for role {role!r}")
    return {"role": role, "limits": [asdict(limit) for limit in limits]}
