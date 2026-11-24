"""Pure policy logic: completed months and eligibility rules (docs/requirements.md 2.1, 2.2)."""

from datetime import date

import pytest

from equipment.domain.models import Basis, EligibilityResult, PolicyLimit, ReasonCode, Status
from equipment.domain.policy import completed_months, evaluate


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


def test_future_start_is_rejected():
    with pytest.raises(ValueError, match="after as_of"):
        completed_months(date(2027, 1, 1), date(2026, 12, 1))


# ---------------------------------------------------------------- evaluate

AS_OF = date(2026, 12, 1)
TWO_PER_24_MONTHS = PolicyLimit(item="monitor", max_qty=2, refresh_months=24)
ONE_PER_24_MONTHS = PolicyLimit(item="laptop", max_qty=1, refresh_months=24)
NOT_ELIGIBLE = PolicyLimit(item="phone", max_qty=0, refresh_months=0)

NEW = date(2026, 7, 1)  # 5 months
OLD = date(2024, 7, 1)  # 29 months
ANNIVERSARY = date(2024, 12, 1)  # exactly 24 months
DAY_BEFORE_ANNIVERSARY = date(2024, 12, 2)  # 23 months
FUTURE = date(2027, 1, 1)


@pytest.mark.parametrize(
    ("limit", "issue_dates", "status", "reason_code"),
    [
        pytest.param(None, [], "needs_review", "RECORDS_INCOMPLETE", id="no-policy"),
        pytest.param(NOT_ELIGIBLE, [], "not_eligible", "ROLE_NOT_ELIGIBLE", id="max-0"),
        pytest.param(NOT_ELIGIBLE, [OLD], "not_eligible", "ROLE_NOT_ELIGIBLE", id="max-0-holds-1"),
        pytest.param(TWO_PER_24_MONTHS, [], "eligible", "WITHIN_LIMIT", id="holds-none"),
        pytest.param(TWO_PER_24_MONTHS, [NEW], "eligible", "WITHIN_LIMIT", id="below-max"),
        pytest.param(TWO_PER_24_MONTHS, [None], "eligible", "WITHIN_LIMIT", id="no-date-below-max"),
        pytest.param(
            TWO_PER_24_MONTHS, [FUTURE], "eligible", "WITHIN_LIMIT", id="future-below-max"
        ),
        pytest.param(
            ONE_PER_24_MONTHS, [None], "needs_review", "RECORDS_INCOMPLETE", id="no-date-at-max"
        ),
        pytest.param(
            ONE_PER_24_MONTHS, [FUTURE], "needs_review", "RECORDS_INCOMPLETE", id="future-at-max"
        ),
        pytest.param(
            TWO_PER_24_MONTHS, [OLD, None], "needs_review", "RECORDS_INCOMPLETE", id="one-no-date"
        ),
        pytest.param(
            ONE_PER_24_MONTHS, [AS_OF], "not_eligible", "REFRESH_NOT_DUE", id="issued-today"
        ),
        pytest.param(ONE_PER_24_MONTHS, [OLD], "eligible", "REFRESH_DUE", id="due"),
        pytest.param(ONE_PER_24_MONTHS, [ANNIVERSARY], "eligible", "REFRESH_DUE", id="anniversary"),
        pytest.param(
            ONE_PER_24_MONTHS,
            [DAY_BEFORE_ANNIVERSARY],
            "not_eligible",
            "REFRESH_NOT_DUE",
            id="day-before-anniversary",
        ),
        pytest.param(
            ONE_PER_24_MONTHS, [NEW, NEW], "not_eligible", "REFRESH_NOT_DUE", id="above-max"
        ),
    ],
)
def test_evaluate(limit, issue_dates, status, reason_code):
    result = evaluate(limit, issue_dates, AS_OF)

    assert (result.status, result.reason_code) == (status, reason_code)


@pytest.mark.parametrize("issue_dates", [[OLD, NEW], [NEW, OLD]])
def test_evaluate_refresh_uses_the_oldest_unit(issue_dates):
    assert evaluate(TWO_PER_24_MONTHS, issue_dates, AS_OF) == EligibilityResult(
        status=Status.ELIGIBLE,
        reason_code=ReasonCode.REFRESH_DUE,
        basis=Basis.REFRESH,
        held=2,
        max_qty=2,
        oldest_age_months=29,
        refresh_months=24,
    )


def test_evaluate_quantity_check_reports_no_age():
    result = evaluate(TWO_PER_24_MONTHS, [OLD], AS_OF)

    assert (result.basis, result.held, result.oldest_age_months) == ("quantity", 1, None)


@pytest.mark.parametrize(
    ("limit", "issue_dates", "missing", "basis", "held"),
    [
        (None, [OLD], "role_policy", None, 1),
        (ONE_PER_24_MONTHS, [None], "issue_date", "refresh", 1),
    ],
)
def test_evaluate_says_what_is_missing(limit, issue_dates, missing, basis, held):
    result = evaluate(limit, issue_dates, AS_OF)

    assert (result.missing, result.basis, result.held) == (missing, basis, held)
