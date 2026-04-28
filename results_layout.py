from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional


RESULTS_ROOT = Path("results")


def sanitize_component(value: object, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text)
    text = re.sub(r"-{2,}", "-", text).strip("-_.")
    return text or fallback


def readable_flare_slug(flare_key: str, flare_class: Optional[str] = None) -> str:
    match = re.match(
        r"^(?P<date>\d{8})T(?P<start>\d{6})_(?P<peak>\d{6})_(?P<end>\d{6})(?:_(?P<class>.+))?$",
        str(flare_key),
    )
    if not match:
        return sanitize_component(flare_key, fallback="flare_unknown")

    date_raw = match.group("date")
    date_label = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"

    def fmt_time(raw: str) -> str:
        return f"{raw[:2]}-{raw[2:4]}-{raw[4:6]}"

    class_label = sanitize_component(flare_class or match.group("class") or "class-unknown")
    return f"{date_label}_{class_label}_{fmt_time(match.group('start'))}_to_{fmt_time(match.group('end'))}"


def event_results_dir(flare_key: str, flare_class: Optional[str] = None, root: Path = RESULTS_ROOT) -> Path:
    return root / readable_flare_slug(flare_key, flare_class=flare_class)


def product_file_name(prefix: str, product: str, flare_key: str, suffix: str, flare_class: Optional[str] = None) -> str:
    return f"{sanitize_component(prefix)}_{sanitize_component(product)}{suffix}"


def source_file_name(source: str, flare_key: str, suffix: str, flare_class: Optional[str] = None) -> str:
    return f"{sanitize_component(source)}{suffix}"


def publish_file(source_path: Path | str, target_path: Path | str, overwrite: bool = False) -> Path:
    source = Path(source_path)
    target = Path(target_path)
    if not source.exists():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == target.resolve():
        return target
    if overwrite or not target.exists() or source.stat().st_mtime > target.stat().st_mtime:
        shutil.copy2(source, target)
    return target
