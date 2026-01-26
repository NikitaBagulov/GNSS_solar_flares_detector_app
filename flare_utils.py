from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from dateutil import tz

DEFAULT_WINDOW_MINUTES = 15


def normalize_to_utc(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))

    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz.UTC)

    return dt.astimezone(tz.UTC)


def build_flare_key(
    start_time,
    peak_time,
    end_time,
    flare_class: Optional[str] = None,
) -> str:
    start = normalize_to_utc(start_time)
    peak = normalize_to_utc(peak_time)
    end = normalize_to_utc(end_time)

    parts = [
        start.strftime("%Y%m%dT%H%M%S"),
        peak.strftime("%H%M%S"),
        end.strftime("%H%M%S"),
    ]

    if flare_class:
        class_tag = re.sub(r"[^A-Za-z0-9]+", "", str(flare_class))
        if class_tag:
            parts.append(class_tag)

    return "_".join(parts)


def get_flare_window(
    start_time,
    end_time,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> Tuple[datetime, datetime]:
    start = normalize_to_utc(start_time) - timedelta(minutes=window_minutes) - timedelta(hours=8)
    end = normalize_to_utc(end_time) + timedelta(minutes=window_minutes) - timedelta(hours=8)
    return start, end
