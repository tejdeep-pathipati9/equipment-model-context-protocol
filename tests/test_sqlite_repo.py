"""SqliteRepository against a fresh seeded temp DB."""

import sqlite3
from datetime import date

import pytest

from equipment.data.seed import EMPLOYEES, EQUIPMENT, POLICIES
from equipment.domain.models import Employee, EquipmentUnit, PolicyLimit


def test_seed_loads_every_row(seeded_conn):
    def count(table):
        return seeded_conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    assert count("employees") == len(EMPLOYEES) == 6
    assert count("equipment") == len(EQUIPMENT)
    assert count("policies") == len(POLICIES) == 20  # 4 roles x 5 items


def test_get_employee_known(repo):
    assert repo.get_employee("E004") == Employee(
        id="E004", name="Dan Okafor", role="manager", hire_date=date(2021, 6, 1)
    )


def test_get_employee_unknown(repo):
    assert repo.get_employee("E999") is None


def test_list_equipment_oldest_first(repo):
    assert repo.list_equipment("E001") == [
        EquipmentUnit(item="laptop", issued_on=date(2024, 9, 1)),
        EquipmentUnit(item="monitor", issued_on=date(2025, 6, 1)),
    ]


def test_list_equipment_none_on_file(repo):
    assert repo.list_equipment("E999") == []


def test_list_equipment_missing_issue_date_sorts_last(repo, seeded_conn):
    seeded_conn.execute(
        "INSERT INTO equipment (employee_id, item, issued_on) VALUES ('E005', 'headset', NULL)"
    )

    assert repo.list_equipment("E005") == [
        EquipmentUnit(item="laptop", issued_on=date(2026, 6, 1)),
        EquipmentUnit(item="headset", issued_on=None),
    ]


def test_list_policies_sorted_by_item(repo):
    assert repo.list_policies("contractor") == [
        PolicyLimit(item="headset", max_qty=1, refresh_months=24),
        PolicyLimit(item="keyboard_mouse", max_qty=1, refresh_months=24),
        PolicyLimit(item="laptop", max_qty=0, refresh_months=0),
        PolicyLimit(item="monitor", max_qty=1, refresh_months=24),
        PolicyLimit(item="phone", max_qty=0, refresh_months=0),
    ]


def test_list_policies_unknown_role(repo):
    assert repo.list_policies("ceo") == []


def test_list_catalog(repo):
    assert repo.list_catalog() == ["headset", "keyboard_mouse", "laptop", "monitor", "phone"]


MANAGER_LAPTOP = "WHERE role = 'manager' AND item = 'laptop'"
E001_LAPTOP = "WHERE employee_id = 'E001' AND item = 'laptop'"


@pytest.mark.parametrize(
    "change",
    [
        pytest.param(f"UPDATE policies SET max_qty = -1 {MANAGER_LAPTOP}", id="negative quantity"),
        pytest.param(
            f"UPDATE policies SET max_qty = 1.5 {MANAGER_LAPTOP}", id="fractional quantity"
        ),
        pytest.param(
            f"UPDATE policies SET max_qty = 'two' {MANAGER_LAPTOP}", id="quantity as text"
        ),
        pytest.param(
            f"UPDATE policies SET refresh_months = -24 {MANAGER_LAPTOP}", id="negative interval"
        ),
        pytest.param(
            f"UPDATE policies SET refresh_months = 0 {MANAGER_LAPTOP}", id="no interval, max 1"
        ),
        pytest.param(
            f"UPDATE policies SET refresh_months = 'soon' {MANAGER_LAPTOP}", id="interval as text"
        ),
        pytest.param(f"UPDATE equipment SET issued_on = 'last May' {E001_LAPTOP}", id="vague date"),
        pytest.param(
            f"UPDATE equipment SET issued_on = '2025-02-30' {E001_LAPTOP}", id="no such day"
        ),
        pytest.param("UPDATE employees SET hire_date = 'soon' WHERE id = 'E001'", id="hire date"),
    ],
)
def test_database_refuses_a_bad_value(seeded_conn, change):
    with pytest.raises(sqlite3.IntegrityError, match="CHECK"):
        seeded_conn.execute(change)
