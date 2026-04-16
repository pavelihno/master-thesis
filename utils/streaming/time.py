from datetime import datetime, timedelta
from enum import Enum
from typing import Any


class TimeTarget(Enum):
    MONTHS = 'months'
    WEEKS = 'weeks'
    DAYS = 'days'
    HOURS = 'hours'
    MINUTES = 'minutes'
    SECONDS = 'seconds'


def parse_time(value: Any) -> datetime | None:
    """Parse event time into datetime.

    Supports:
    - datetime objects (returned as-is)
    - ISO-like strings (including trailing 'Z')
    - UNIX timestamps (int/float)
    """
    if value is None:
        return None

    if isinstance(value, datetime):
        return value

    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value)
        except (OverflowError, OSError, ValueError):
            return None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None

        # Handle common UTC suffix in ISO strings.
        if text.endswith('Z'):
            text = f'{text[:-1]}+00:00'

        try:
            return datetime.fromisoformat(text)
        except ValueError:
            return None

    return None


def convert_time(
    delta: timedelta | float | int | None,
    target: TimeTarget = TimeTarget.SECONDS,
) -> float:
    """Convert a time delta to a selected unit."""
    unit = TimeTarget(target)

    if delta is None:
        return 0.0

    if isinstance(delta, timedelta):
        total_seconds = delta.total_seconds()
    else:
        total_seconds = float(delta)

    factors = {
        TimeTarget.MONTHS: 30 * 24 * 60 * 60,
        TimeTarget.WEEKS: 7 * 24 * 60 * 60,
        TimeTarget.DAYS: 24 * 60 * 60,
        TimeTarget.HOURS: 60 * 60,
        TimeTarget.MINUTES: 60,
        TimeTarget.SECONDS: 1,
    }

    return total_seconds / factors[unit]
