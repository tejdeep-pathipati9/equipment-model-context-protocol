"""Pure policy logic. All date logic takes an explicit `as_of` date (docs/requirements.md §2.2)."""

import calendar
from datetime import date


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
