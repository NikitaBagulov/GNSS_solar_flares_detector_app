"""Identify catalog metadata for an event without guessing a flare position."""

import csv
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def _utc(value):
    try:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
        return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)
    except (ValueError, TypeError):
        return None


def flare_metadata(event: Path) -> dict:
    match = re.match(r"^(\d{4}-\d{2}-\d{2})_([ABCMX]\d+(?:\.\d+)?)$", event.name, re.I)
    if not match:
        return {"class": "", "date": "", "start": "", "peak": "", "end": "", "x": None, "y": None}
    date, flare_class = match.groups()
    result = {"class": flare_class.upper(), "date": date, "start": "", "peak": "", "end": "", "x": None, "y": None}
    root = Path(__file__).resolve().parent / "data"
    candidates = []
    for filename, class_column, start_column, peak_column, end_column in (
        ("all_flares.csv", "class", "start_time", "peak_time", "end_time"),
        ("flare_position_catalog_hek.csv", "fl_goescls", "event_starttime", "event_peaktime", "event_endtime"),
    ):
        path = root / filename
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or not {class_column, start_column, peak_column}.issubset(reader.fieldnames):
                    continue
                for row in reader:
                    start = _utc(row.get(start_column))
                    if start and start.date().isoformat() == date and str(row.get(class_column, "")).strip().upper() == result["class"]:
                        candidates.append((filename, row, start, _utc(row.get(peak_column)), _utc(row.get(end_column))))
        except OSError:
            continue
    # Prefer the processing catalog; the HEK catalog is a fallback, not a second flare.
    hek_candidates = [item for item in candidates if item[0] == "flare_position_catalog_hek.csv"]
    if any(item[0] == "all_flares.csv" for item in candidates):
        candidates = [item for item in candidates if item[0] == "all_flares.csv"]
    # Multiple flares with the same class on a day cannot be identified by the folder name alone.
    # Only use a candidate if the stored map epochs distinguish its observation window.
    if len(candidates) > 1:
        try:
            from plot_requests import map_epochs
            stamps = [_utc(key) for keys in map_epochs(event).values() for key in keys]
            stamps = [stamp for stamp in stamps if stamp]
            if stamps:
                low, high = min(stamps), max(stamps)
                candidates = [item for item in candidates if item[3] and low <= item[3] <= high]
        except (OSError, ValueError):
            candidates = []
    if len(candidates) == 1:
        _, row, start, peak, end = candidates[0]
        result.update(start=start.isoformat(timespec="minutes"),
                      peak=peak.isoformat(timespec="minutes") if peak else "",
                      end=end.isoformat(timespec="minutes") if end else "")
        position_rows = [row]
        if peak:
            position_rows.extend(item[1] for item in hek_candidates if item[3] and
                                 abs((item[3] - peak).total_seconds()) <= 120)
        for position_row in position_rows:
            try:
                x, y = float(position_row["hpc_x"]), float(position_row["hpc_y"])
                if math.isfinite(x) and math.isfinite(y) and x*x + y*y <= 960*960:
                    result.update(x=x, y=y)
                    break
            except (KeyError, TypeError, ValueError):
                continue
    return result


def scientific_caption(metadata: dict, series=()) -> tuple[str, str]:
    """Describe the measured phenomenon rather than the event's directory name."""
    flare = f"the {metadata['class']} solar flare" if metadata["class"] else "a solar flare"
    if any(key.startswith("map:") or ":" in key for key in series):
        title = f"Ionospheric response to {flare}"
    elif any(key in ("goes", "soho") for key in series):
        title = f"Solar irradiance during {flare}"
    else:
        title = f"Solar observations of {flare}"
    details = []
    if metadata["date"]:
        try:
            date = datetime.strptime(metadata["date"], "%Y-%m-%d")
            details.append(f"{date.day} {date:%B %Y}")
        except ValueError:
            details.append(metadata["date"])
    for label, field in (("Start", "start"), ("Peak", "peak"), ("End", "end")):
        if metadata[field]:
            details.append(f"{label} {metadata[field][11:16]} UTC")
    if metadata["x"] is not None:
        details.append(f"HPC ({metadata['x']:.0f}\u2033, {metadata['y']:.0f}\u2033)")
    return title, "  ·  ".join(details)
