"""Pure policy logic. All date logic takes an explicit `as_of` date (docs/requirements.md §2.2)."""

import calendar
from datetime import date

from equipment.domain.models import Basis, EligibilityResult, PolicyLimit, ReasonCode, Status


def completed_months(start: date, as_of: date) -> int:
    """Completed calendar months from `start` to `as_of`.

    A month completes on the same day-of-month as `start`; if that day doesn't exist in the
    month, it completes on the month's last day. A `start` after `as_of` is rejected: callers
    check for future dates explicitly (they are unreliable data, not a negative age).
    """
    if start > as_of:
        raise ValueError(f"start {start} is after as_of {as_of}")
    months = (as_of.year - start.year) * 12 + (as_of.month - start.month)
    _, days_in_month = calendar.monthrange(as_of.year, as_of.month)
    if as_of.day < start.day and as_of.day != days_in_month:
        months -= 1
    return months


def evaluate(
    limit: PolicyLimit | None, issue_dates: list[date | None], as_of: date
) -> EligibilityResult:
    """Eligibility rules 2-7 for one item; `issue_dates` has one entry per unit held.

    Example: max 2, refresh 24, units aged 29 and 5 months -> REFRESH_DUE (oldest is 29).
    """
    held = len(issue_dates)
    if limit is None:
        return EligibilityResult(
            Status.NEEDS_REVIEW, ReasonCode.RECORDS_INCOMPLETE, held=held, missing="role_policy"
        )

    def result(status: Status, reason_code: ReasonCode, **details) -> EligibilityResult:
        return EligibilityResult(
            status,
            reason_code,
            held=held,
            max_qty=limit.max_qty,
            refresh_months=limit.refresh_months,
            **details,
        )

    if limit.max_qty == 0:
        return result(Status.NOT_ELIGIBLE, ReasonCode.ROLE_NOT_ELIGIBLE)
    if held < limit.max_qty:
        return result(Status.ELIGIBLE, ReasonCode.WITHIN_LIMIT, basis=Basis.QUANTITY)

    # At max: the only question left is whether the oldest unit is due for replacement.
    reliable = [issued for issued in issue_dates if issued is not None and issued <= as_of]
    if len(reliable) < held:
        return result(
            Status.NEEDS_REVIEW,
            ReasonCode.RECORDS_INCOMPLETE,
            basis=Basis.REFRESH,
            missing="issue_date",
        )

    age = completed_months(min(reliable), as_of)
    if age >= limit.refresh_months:
        status, reason_code = Status.ELIGIBLE, ReasonCode.REFRESH_DUE
    else:
        status, reason_code = Status.NOT_ELIGIBLE, ReasonCode.REFRESH_NOT_DUE
    return result(status, reason_code, basis=Basis.REFRESH, oldest_age_months=age)
