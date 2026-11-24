"""Pure policy logic. All date logic takes an explicit `as_of` date (docs/requirements.md §2.2)."""

import calendar
from datetime import date


def completed_months(start: date, as_of: date) -> int:
    """Completed calendar months from `start` to `as_of`.

    A month completes on the same day-of-month as `start`; if that day doesn't exist in the
    month, it completes on the month's last day. Negative if `start` is after `as_of`.

    >>> completed_months(date(2024, 12, 1), date(2026, 12, 1))
    24
    >>> completed_months(date(2024, 12, 1), date(2026, 11, 30))
    23
    >>> completed_months(date(2025, 1, 31), date(2025, 2, 28))
    1
    """
    if start > as_of:
        return -completed_months(as_of, start)
    months = (as_of.year - start.year) * 12 + (as_of.month - start.month)
    last_day_of_month = calendar.monthrange(as_of.year, as_of.month)[1]
    if as_of.day < start.day and as_of.day != last_day_of_month:
        months -= 1
    return months
