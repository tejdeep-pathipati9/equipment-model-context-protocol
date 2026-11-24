"""Domain vocabulary stays consistent with the seed data and the requirements."""

from equipment.data.seed import EMPLOYEES, EQUIPMENT, POLICIES
from equipment.domain.models import Item, ReasonCode, Role


def test_seed_roles_are_known_roles():
    assert {role for _, _, role, _ in EMPLOYEES} <= set(Role)


def test_seed_items_are_catalog_items():
    assert {item for _, item, _ in EQUIPMENT} <= set(Item)


def test_policy_table_covers_every_role_and_item():
    assert {(role, item) for role, item, _, _ in POLICIES} == {(r, i) for r in Role for i in Item}


def test_enum_values_compare_as_plain_strings():
    # StrEnum lets tool results serialize to JSON as plain strings
    assert Item.MONITOR == "monitor"
    assert ReasonCode.WITHIN_LIMIT == "WITHIN_LIMIT"
