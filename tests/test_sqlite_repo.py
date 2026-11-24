"""SqliteRepository against a fresh seeded temp DB."""

from datetime import date

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
