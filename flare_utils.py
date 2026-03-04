from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
from dateutil import tz
import pandas as pd

DEFAULT_WINDOW_MINUTES = 15
_UTC = tz.gettz('UTC')

def normalize_to_utc(value) -> datetime:
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value))

    if dt.tzinfo is None:
        return dt.replace(tzinfo=tz.UTC)

    return dt


def build_flare_key(
    start_time,
    peak_time,
    end_time,
    flare_class: Optional[str] = None,
) -> str:
    start = start_time
    peak = peak_time
    end = end_time

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


def _as_utc_timestamp(dt_like: object) -> pd.Timestamp:
    """
    Приводим любое время (datetime / pd.Timestamp / строку) к pd.Timestamp в UTC (tz-aware).
    """
    t = pd.Timestamp(dt_like)

    # Если tz нет — считаем, что это уже UTC, просто "помечаем"
    if t.tz is None:
        t = t.tz_localize("UTC")
    else:
        # Если tz есть — конвертируем в UTC, сохраняя момент времени
        t = t.tz_convert("UTC")

    return t


def get_flare_window(
    start_time: datetime,
    end_time: datetime,
    window_minutes: int = DEFAULT_WINDOW_MINUTES,
) -> Tuple[datetime, datetime]:
    start_utc = _as_utc_timestamp(start_time).to_pydatetime()
    end_utc = _as_utc_timestamp(end_time).to_pydatetime()

    start = start_utc - timedelta(minutes=window_minutes)
    end = end_utc + timedelta(minutes=window_minutes)

    return start, end
