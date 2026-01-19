from __future__ import annotations

import re
from datetime import timedelta
from typing import Optional, Tuple

import pandas as pd


DEFAULT_WINDOW_MINUTES = 15


def normalize_to_utc(value) -> pd.Timestamp:
    return pd.to_datetime(value, utc=True)


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
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = normalize_to_utc(start_time) - timedelta(minutes=window_minutes)
    end = normalize_to_utc(end_time) + timedelta(minutes=window_minutes)
    return start, end
