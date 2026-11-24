"""Pure policy logic: completed calendar months (docs/requirements.md §2.2)."""

from datetime import date

import pytest

from equipment.domain.policy import completed_months


@pytest.mark.parametrize(
    ("start", "as_of", "expected"),
    [
        pytest.param(date(2024, 12, 1), date(2026, 12, 1), 24, id="on-anniversary"),
        pytest.param(date(2024, 12, 1), date(2026, 11, 30), 23, id="day-before-anniversary"),
        pytest.param(date(2026, 12, 1), date(2026, 12, 1), 0, id="same-day"),
        pytest.param(date(2025, 1, 31), date(2025, 2, 28), 1, id="month-end-feb"),
        pytest.param(date(2025, 1, 31), date(2025, 2, 27), 0, id="before-month-end-feb"),
        pytest.param(date(2024, 1, 31), date(2024, 2, 29), 1, id="month-end-leap-year"),
        pytest.param(date(2025, 3, 31), date(2025, 4, 30), 1, id="month-end-30-day-month"),
        pytest.param(date(2023, 5, 1), date(2026, 12, 1), 43, id="seed-E006-laptop"),
        pytest.param(date(2026, 3, 15), date(2026, 12, 1), 8, id="seed-E003-monitors"),
        pytest.param(date(2025, 12, 1), date(2026, 12, 1), 12, id="seed-E004-laptop"),
    ],
)
def test_completed_months(start, as_of, expected):
    assert completed_months(start, as_of) == expected


def test_future_start_is_negative():
    assert completed_months(date(2027, 1, 1), date(2026, 12, 1)) == -1
