from __future__ import annotations

import argparse
import html
import json
import os
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


FILE_TYPE_LABELS = {
    ".csv": "CSV",
    ".h5": "HDF5",
    ".hdf5": "HDF5",
    ".hdf": "HDF",
    ".json": "JSON",
    ".png": "PNG",
    ".jpg": "JPG",
    ".jpeg": "JPEG",
    ".txt": "TXT",
}
PRODUCTS = ("roti", "dtec_2_10", "dtec_10_20", "dtec_20_60")
SOURCE_FILES = ("goes_xray.csv", "soho_sem.csv")
GRAPH_PRODUCTS = (*PRODUCTS, "combined")


def relative_url(root: Path, path: Path) -> str:
    return "/" + quote(path.relative_to(root).as_posix())


def safe_stat(path: Path):
    try:
        return path.stat()
    except OSError:
        return None


def format_size(size: int | None) -> str:
    if size is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def parse_size_label(label: str) -> float:
    if not label or label == "-":
        return 0.0
    value, _, unit = label.partition(" ")
    try:
        number = float(value)
    except ValueError:
        return 0.0
    unit = unit.upper()
    powers = {"B": 0, "KB": 1, "MB": 2, "GB": 3, "TB": 4}
    return number * (1024 ** powers.get(unit, 0))


def file_kind(path: Path) -> str:
    if path.is_dir():
        return "Folder"
    return FILE_TYPE_LABELS.get(path.suffix.lower(), path.suffix[1:].upper() or "File")


def dir_size(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            stat = safe_stat(child)
            if stat:
                total += stat.st_size
    return total


def count_files(path: Path, pattern: str) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.rglob(pattern) if item.is_file())


def first_png(path: Path) -> Path | None:
    if not path.exists():
        return None
    for item in path.rglob("*.png"):
        if item.is_file():
            return item
    return None


def graph_product(path: Path) -> str:
    parent = path.parent.name
    if parent in GRAPH_PRODUCTS:
        return parent
    name = path.name.lower()
    if name.startswith("combined_"):
        return "combined"
    for product in PRODUCTS:
        if f"_{product}_" in name or name.startswith(f"map_{product}_"):
            return product
    return parent


def graph_time_label(path: Path) -> str:
    stem = path.stem
    if "_UTC" in stem:
        stem = stem.rsplit("_UTC", 1)[0]
    tail = stem.rsplit("_", 1)[-1]
    hyphen_parts = tail.split("-")
    if len(hyphen_parts) == 3 and all(part.isdigit() for part in hyphen_parts):
        return ":".join(hyphen_parts)
    parts = stem.split("_")
    if len(parts) >= 3 and parts[-3].isdigit() and parts[-2].isdigit() and parts[-1].isdigit():
        return ":".join(parts[-3:])
    return stem


def is_graphs_dir(root: Path, path: Path) -> bool:
    if not path.is_dir():
        return False
    try:
        relative_parts = path.relative_to(root).parts
    except ValueError:
        return False
    return "graphs" in relative_parts and first_png(path) is not None


def scan_graph_images(root: Path, path: Path) -> list[dict]:
    images = []
    for image in sorted(path.rglob("*.png")):
        if not image.is_file():
            continue
        stat = safe_stat(image)
        images.append(
            {
                "name": image.name,
                "path": image.relative_to(root).as_posix(),
                "url": relative_url(root, image),
                "product": graph_product(image),
                "time": graph_time_label(image),
                "size": format_size(stat.st_size if stat else None),
                "modified": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M") if stat else "",
            }
        )
    return images


def newest_mtime(path: Path) -> float:
    newest = 0.0
    for item in path.rglob("*"):
        stat = safe_stat(item)
        if stat and stat.st_mtime > newest:
            newest = stat.st_mtime
    stat = safe_stat(path)
    if stat and stat.st_mtime > newest:
        newest = stat.st_mtime
    return newest


def is_event_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    if path.name in {"maps", "indices", "graphs", "goes_xray", "soho_sem", "combined", *PRODUCTS}:
        return False
    names = {child.name for child in path.iterdir()} if path.exists() else set()
    return bool(names & {"maps", "indices", "graphs", "goes_xray", "soho_sem"})


def event_class(path: Path) -> str:
    parent = path.parent.name.upper()
    if parent in {"A", "B", "C", "M", "X"}:
        return parent
    if "_" in path.name:
        return path.name.rsplit("_", 1)[-1].upper()[:1]
    return "?"


def event_date(path: Path) -> str:
    name = path.name
    if len(name) >= 10 and name[4:5] == "-" and name[7:8] == "-":
        return name[:10]
    if len(name) >= 8 and name[:8].isdigit():
        return f"{name[:4]}-{name[4:6]}-{name[6:8]}"
    return ""


def scan_event(root: Path, path: Path) -> dict:
    maps_dir = path / "maps"
    indices_dir = path / "indices"
    graphs_dir = path / "graphs"
    source_status = {
        source.removesuffix(".csv"): (path / source.removesuffix(".csv") / source).exists()
        or (path / source).exists()
        for source in SOURCE_FILES
    }
    maps = {product: (maps_dir / f"map_{product}.h5").exists() for product in PRODUCTS}
    indices = {product: (indices_dir / f"indices_{product}.csv").exists() for product in PRODUCTS}
    preview = first_png(graphs_dir)
    size = dir_size(path)
    complete_checks = [
        sum(maps.values()) == len(PRODUCTS),
        sum(indices.values()) == len(PRODUCTS),
        count_files(graphs_dir, "*.png") > 0,
        source_status.get("goes_xray", False),
        source_status.get("soho_sem", False),
    ]
    return {
        "name": path.name,
        "path": path.relative_to(root).as_posix(),
        "url": relative_url(root, path) + "/",
        "date": event_date(path),
        "class": event_class(path),
        "size_bytes": size,
        "size": format_size(size),
        "modified": datetime.fromtimestamp(newest_mtime(path)).strftime("%Y-%m-%d %H:%M"),
        "maps_ready": sum(maps.values()),
        "maps_total": len(PRODUCTS),
        "indices_ready": sum(indices.values()),
        "indices_total": len(PRODUCTS),
        "graphs_count": count_files(graphs_dir, "*.png"),
        "combined_count": count_files(graphs_dir / "combined", "*.png"),
        "sources": source_status,
        "maps": maps,
        "indices": indices,
        "complete": all(complete_checks),
        "preview_url": relative_url(root, preview) if preview else None,
    }


def scan_events(root: Path) -> list[dict]:
    if not root.exists():
        return []
    events = []
    class_dirs = {"A", "B", "C", "M", "X", "unknown"}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if child.name in class_dirs:
            for event_dir in sorted(item for item in child.iterdir() if item.is_dir()):
                if is_event_dir(event_dir):
                    events.append(scan_event(root, event_dir))
        elif is_event_dir(child):
            events.append(scan_event(root, child))
    return sorted(events, key=lambda event: (event["date"], event["name"]))


def build_summary(events: list[dict]) -> dict:
    total_size = sum(event["size_bytes"] for event in events)
    return {
        "events": len(events),
        "complete": sum(1 for event in events if event["complete"]),
        "incomplete": sum(1 for event in events if not event["complete"]),
        "with_graphs": sum(1 for event in events if event["graphs_count"] > 0),
        "missing_soho_sem": sum(1 for event in events if not event["sources"].get("soho_sem")),
        "missing_goes_xray": sum(1 for event in events if not event["sources"].get("goes_xray")),
        "size_bytes": total_size,
        "size": format_size(total_size),
    }


def entry_sort_key(path: Path):
    return (not path.is_dir(), path.name.lower())


def breadcrumb_items(url_path: str):
    clean_path = unquote(url_path).strip("/")
    parts = [part for part in clean_path.split("/") if part]
    items = [("Results", "/")]
    current = ""
    for part in parts:
        current += "/" + quote(part)
        items.append((part, current + "/"))
    return items


def render_directory_html(url_path: str, entries: list[Path]) -> bytes:
    title_path = unquote(url_path).strip("/") or "results"
    directories = [entry for entry in entries if entry.is_dir()]
    files = [entry for entry in entries if entry.is_file()]
    total_size = sum(entry.stat().st_size for entry in files)
    rows = []

    if url_path != "/":
        rows.append(
            """
            <tr>
              <td><a class="name-link parent-link" href="../">..</a></td>
              <td><span class="badge folder">Parent</span></td>
              <td class="muted">-</td>
              <td class="muted">-</td>
            </tr>
            """
        )

    for entry in sorted(entries, key=entry_sort_key):
        stat = entry.stat()
        name = entry.name + ("/" if entry.is_dir() else "")
        href = quote(entry.name) + ("/" if entry.is_dir() else "")
        kind = file_kind(entry)
        badge_class = "folder" if entry.is_dir() else "file"
        modified = datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        size = "-" if entry.is_dir() else format_size(stat.st_size)
        rows.append(
            f"""
            <tr>
              <td><a class="name-link" href="{href}">{html.escape(name)}</a></td>
              <td><span class="badge {badge_class}">{html.escape(kind)}</span></td>
              <td>{html.escape(size)}</td>
              <td>{html.escape(modified)}</td>
            </tr>
            """
        )

    breadcrumbs = " / ".join(
        f'<a href="{href}">{html.escape(label)}</a>'
        for label, href in breadcrumb_items(url_path)
    )

    empty_state = ""
    if not rows:
        empty_state = '<tr><td colspan="4" class="empty">No files in this folder.</td></tr>'

    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Results - {html.escape(title_path)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --text: #18202a;
      --muted: #687383;
      --line: #dde3ea;
      --accent: #0f766e;
      --accent-soft: #dff3ef;
      --file-soft: #eef2f7;
      --shadow: 0 12px 32px rgba(24, 32, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 "Segoe UI", Arial, sans-serif;
    }}
    main {{
      width: min(1180px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 44px;
    }}
    header {{
      display: flex;
      justify-content: space-between;
      gap: 20px;
      align-items: flex-end;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 6px;
      font-size: 28px;
      line-height: 1.2;
      letter-spacing: 0;
    }}
    .breadcrumbs, .breadcrumbs a {{
      color: var(--muted);
      text-decoration: none;
    }}
    .breadcrumbs a:hover {{ color: var(--accent); }}
    .stats {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      justify-content: flex-end;
    }}
    .stat {{
      min-width: 104px;
      padding: 10px 12px;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: 0 4px 16px rgba(24, 32, 42, 0.04);
    }}
    .stat strong {{
      display: block;
      font-size: 18px;
      line-height: 1.1;
    }}
    .stat span {{
      color: var(--muted);
      font-size: 12px;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      box-shadow: var(--shadow);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 680px;
    }}
    th, td {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      text-align: left;
      white-space: nowrap;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    tr:last-child td {{ border-bottom: 0; }}
    tbody tr:hover {{ background: #f9fbfc; }}
    .name-link {{
      color: var(--text);
      font-weight: 600;
      text-decoration: none;
    }}
    .name-link:hover {{ color: var(--accent); }}
    .parent-link {{ color: var(--muted); }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-width: 68px;
      justify-content: center;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 600;
    }}
    .badge.folder {{
      color: #075e57;
      background: var(--accent-soft);
    }}
    .badge.file {{
      color: #475569;
      background: var(--file-soft);
    }}
    .muted, .empty {{ color: var(--muted); }}
    .empty {{
      padding: 34px 16px;
      text-align: center;
    }}
    @media (max-width: 720px) {{
      main {{ width: min(100vw - 20px, 1180px); padding-top: 18px; }}
      header {{ display: block; }}
      h1 {{ font-size: 22px; }}
      .stats {{ justify-content: flex-start; margin-top: 14px; }}
      .stat {{ min-width: 92px; }}
    }}
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>{html.escape(title_path)}</h1>
        <div class="breadcrumbs">{breadcrumbs}</div>
      </div>
      <div class="stats">
        <div class="stat"><strong>{len(directories)}</strong><span>folders</span></div>
        <div class="stat"><strong>{len(files)}</strong><span>files</span></div>
        <div class="stat"><strong>{html.escape(format_size(total_size))}</strong><span>file size</span></div>
      </div>
    </header>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Size</th>
            <th>Modified</th>
          </tr>
        </thead>
        <tbody>
          {''.join(rows) or empty_state}
        </tbody>
      </table>
    </div>
  </main>
</body>
</html>
"""
    return html_doc.encode("utf-8", "surrogateescape")


def page_shell(title: str, body: str, extra_head: str = "") -> bytes:
    html_doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --text: #18202a;
      --muted: #687383;
      --line: #dde3ea;
      --accent: #0f766e;
      --good: #0f766e;
      --bad: #b42318;
      --warn: #b45309;
      --soft: #eef2f7;
      --shadow: 0 12px 32px rgba(24, 32, 42, 0.08);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.5 "Segoe UI", Arial, sans-serif;
    }}
    main {{ width: min(1320px, calc(100vw - 32px)); margin: 0 auto; padding: 28px 0 44px; }}
    header {{ display: flex; justify-content: space-between; gap: 20px; align-items: flex-end; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 6px; font-size: 28px; line-height: 1.2; letter-spacing: 0; }}
    h2 {{ margin: 28px 0 12px; font-size: 18px; }}
    a {{ color: inherit; }}
    .muted, .breadcrumbs, .breadcrumbs a {{ color: var(--muted); text-decoration: none; }}
    .breadcrumbs a:hover {{ color: var(--accent); }}
    .stats {{ display: flex; gap: 10px; flex-wrap: wrap; justify-content: flex-end; }}
    .stat {{ min-width: 112px; padding: 10px 12px; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; box-shadow: 0 4px 16px rgba(24,32,42,.04); }}
    .stat strong {{ display: block; font-size: 18px; line-height: 1.1; }}
    .stat span {{ color: var(--muted); font-size: 12px; }}
    .toolbar {{ display: grid; grid-template-columns: minmax(180px, 1fr) 150px 160px 180px; gap: 10px; margin: 18px 0; }}
    .toolbar input, .toolbar select {{ width: 100%; padding: 9px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); color: var(--text); }}
    .table-wrap {{ overflow-x: auto; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; box-shadow: var(--shadow); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 860px; }}
    th, td {{ padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; white-space: nowrap; }}
    th {{ color: var(--muted); font-size: 12px; font-weight: 600; text-transform: uppercase; }}
    tbody tr:hover {{ background: #f9fbfc; }}
    .name-link {{ color: var(--text); font-weight: 600; text-decoration: none; }}
    .name-link:hover {{ color: var(--accent); }}
    .badge {{ display: inline-flex; align-items: center; min-width: 42px; justify-content: center; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 600; background: var(--soft); color: #475569; }}
    .ok {{ color: var(--good); background: #dff3ef; }}
    .bad {{ color: var(--bad); background: #fee4e2; }}
    .warn {{ color: var(--warn); background: #fef0c7; }}
    .cards {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); gap: 12px; }}
    .card {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; box-shadow: 0 4px 16px rgba(24,32,42,.04); }}
    .thumb {{ display: block; width: 100%; aspect-ratio: 16 / 9; object-fit: cover; background: #e7ebf0; }}
    .card-body {{ padding: 12px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 10px; }}
    .panel {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .kv {{ display: flex; justify-content: space-between; gap: 12px; border-bottom: 1px solid var(--line); padding: 7px 0; }}
    .kv:last-child {{ border-bottom: 0; }}
    .actions {{ display: flex; gap: 8px; flex-wrap: wrap; margin-top: 12px; }}
    .button {{ display: inline-flex; align-items: center; padding: 8px 10px; border: 1px solid var(--line); border-radius: 8px; text-decoration: none; background: var(--surface); }}
    .button:hover {{ border-color: var(--accent); color: var(--accent); }}
    @media (max-width: 840px) {{
      main {{ width: min(100vw - 20px, 1320px); padding-top: 18px; }}
      header {{ display: block; }}
      h1 {{ font-size: 22px; }}
      .stats {{ justify-content: flex-start; margin-top: 14px; }}
      .toolbar {{ grid-template-columns: 1fr; }}
    }}
  </style>
  {extra_head}
</head>
<body><main>{body}</main></body>
</html>
"""
    return html_doc.encode("utf-8", "surrogateescape")


def render_dashboard(root: Path) -> bytes:
    events = scan_events(root)
    summary = build_summary(events)
    recent = sorted(events, key=lambda event: event["modified"], reverse=True)[:12]
    rows = []
    for event in events:
        rows.append(
            f"""<tr data-name="{html.escape(event['name'].lower())}" data-class="{html.escape(event['class'])}" data-complete="{str(event['complete']).lower()}" data-size="{event['size_bytes']}">
              <td><a class="name-link" href="{html.escape(event['url'])}">{html.escape(event['name'])}</a></td>
              <td>{html.escape(event['date'])}</td>
              <td><span class="badge">{html.escape(event['class'])}</span></td>
              <td><span class="badge {'ok' if event['maps_ready'] == event['maps_total'] else 'warn'}">{event['maps_ready']}/{event['maps_total']}</span></td>
              <td><span class="badge {'ok' if event['indices_ready'] == event['indices_total'] else 'warn'}">{event['indices_ready']}/{event['indices_total']}</span></td>
              <td><span class="badge {'ok' if event['graphs_count'] else 'bad'}">{event['graphs_count']}</span></td>
              <td><span class="badge {'ok' if event['sources'].get('soho_sem') else 'bad'}">SOHO</span></td>
              <td><span class="badge {'ok' if event['sources'].get('goes_xray') else 'bad'}">GOES</span></td>
              <td>{html.escape(event['size'])}</td>
              <td>{html.escape(event['modified'])}</td>
            </tr>"""
        )

    cards = []
    for event in recent:
        preview = event["preview_url"]
        thumb = (
            f'<img class="thumb" src="{html.escape(preview)}" alt="">'
            if preview
            else '<div class="thumb"></div>'
        )
        cards.append(
            f"""<article class="card">
              <a href="{html.escape(event['url'])}">{thumb}</a>
              <div class="card-body">
                <a class="name-link" href="{html.escape(event['url'])}">{html.escape(event['name'])}</a>
                <div class="muted">{html.escape(event['modified'])} · {html.escape(event['size'])}</div>
              </div>
            </article>"""
        )

    body = f"""
    <header>
      <div>
        <h1>GNSS Solar Flare Results</h1>
        <div class="breadcrumbs"><a href="/">Results</a> / <a href="/api/summary">API summary</a> / <a href="/api/events">API events</a></div>
      </div>
      <div class="stats">
        <div class="stat"><strong>{summary['events']}</strong><span>events</span></div>
        <div class="stat"><strong>{summary['complete']}</strong><span>complete</span></div>
        <div class="stat"><strong>{summary['incomplete']}</strong><span>incomplete</span></div>
        <div class="stat"><strong>{summary['missing_soho_sem']}</strong><span>missing SOHO</span></div>
        <div class="stat"><strong>{html.escape(summary['size'])}</strong><span>total size</span></div>
      </div>
    </header>
    <div class="toolbar">
      <input id="q" placeholder="Search date, class, name">
      <select id="classFilter"><option value="">All classes</option><option value="X">X</option><option value="M">M</option><option value="C">C</option></select>
      <select id="statusFilter"><option value="">All statuses</option><option value="complete">Complete</option><option value="incomplete">Incomplete</option></select>
      <select id="sortBy"><option value="date">Sort by date</option><option value="name">Sort by name</option><option value="size">Sort by size</option></select>
    </div>
    <div class="table-wrap">
      <table id="eventsTable">
        <thead><tr><th>Name</th><th>Date</th><th>Class</th><th>Maps</th><th>Indices</th><th>Graphs</th><th>SOHO</th><th>GOES</th><th>Size</th><th>Modified</th></tr></thead>
        <tbody>{''.join(rows)}</tbody>
      </table>
    </div>
    <h2>Recently Updated</h2>
    <div class="cards">{''.join(cards)}</div>
    <script>
      const q = document.getElementById('q');
      const classFilter = document.getElementById('classFilter');
      const statusFilter = document.getElementById('statusFilter');
      const sortBy = document.getElementById('sortBy');
      const tbody = document.querySelector('#eventsTable tbody');
      const originalRows = Array.from(tbody.querySelectorAll('tr'));
      function applyFilters() {{
        const query = q.value.toLowerCase();
        const klass = classFilter.value;
        const status = statusFilter.value;
        let rows = originalRows.filter(row => {{
          const text = row.innerText.toLowerCase();
          const okQuery = !query || text.includes(query);
          const okClass = !klass || row.dataset.class === klass;
          const okStatus = !status || (status === 'complete') === (row.dataset.complete === 'true');
          return okQuery && okClass && okStatus;
        }});
        rows.sort((a, b) => {{
          if (sortBy.value === 'size') return Number(b.dataset.size) - Number(a.dataset.size);
          if (sortBy.value === 'name') return a.dataset.name.localeCompare(b.dataset.name);
          return a.children[1].innerText.localeCompare(b.children[1].innerText);
        }});
        tbody.replaceChildren(...rows);
      }}
      [q, classFilter, statusFilter, sortBy].forEach(el => el.addEventListener('input', applyFilters));
    </script>
    """
    return page_shell("GNSS Results", body)


def render_event_page(root: Path, path: Path) -> bytes:
    event = scan_event(root, path)
    breadcrumbs = " / ".join(
        f'<a href="{href}">{html.escape(label)}</a>'
        for label, href in breadcrumb_items("/" + path.relative_to(root).as_posix() + "/")
    )
    product_rows = []
    for product in PRODUCTS:
        product_rows.append(
            f"""<tr>
              <td>{html.escape(product)}</td>
              <td><span class="badge {'ok' if event['maps'][product] else 'bad'}">{'yes' if event['maps'][product] else 'no'}</span></td>
              <td><span class="badge {'ok' if event['indices'][product] else 'bad'}">{'yes' if event['indices'][product] else 'no'}</span></td>
            </tr>"""
        )
    preview = event["preview_url"]
    preview_html = (
        f'<a href="{html.escape(preview)}"><img class="thumb" src="{html.escape(preview)}" alt=""></a>'
        if preview
        else '<div class="thumb"></div>'
    )
    body = f"""
    <header>
      <div>
        <h1>{html.escape(event['name'])}</h1>
        <div class="breadcrumbs">{breadcrumbs}</div>
      </div>
      <div class="stats">
        <div class="stat"><strong>{event['maps_ready']}/{event['maps_total']}</strong><span>maps</span></div>
        <div class="stat"><strong>{event['indices_ready']}/{event['indices_total']}</strong><span>indices</span></div>
        <div class="stat"><strong>{event['graphs_count']}</strong><span>graphs</span></div>
        <div class="stat"><strong>{html.escape(event['size'])}</strong><span>size</span></div>
      </div>
    </header>
    <div class="grid">
      <section class="panel">
        <h2>Event Status</h2>
        <div class="kv"><span>Date</span><strong>{html.escape(event['date'])}</strong></div>
        <div class="kv"><span>Class</span><strong>{html.escape(event['class'])}</strong></div>
        <div class="kv"><span>SOHO SEM</span><span class="badge {'ok' if event['sources'].get('soho_sem') else 'bad'}">{'present' if event['sources'].get('soho_sem') else 'missing'}</span></div>
        <div class="kv"><span>GOES X-ray</span><span class="badge {'ok' if event['sources'].get('goes_xray') else 'bad'}">{'present' if event['sources'].get('goes_xray') else 'missing'}</span></div>
        <div class="actions">
          <a class="button" href="maps/">maps</a>
          <a class="button" href="indices/">indices</a>
          <a class="button" href="graphs/">graphs</a>
          <a class="button" href="graphs/combined/">combined</a>
        </div>
      </section>
      <section class="panel">
        <h2>Preview</h2>
        {preview_html}
      </section>
    </div>
    <h2>Products</h2>
    <div class="table-wrap">
      <table><thead><tr><th>Product</th><th>Map</th><th>Index</th></tr></thead><tbody>{''.join(product_rows)}</tbody></table>
    </div>
    <h2>Files</h2>
    """
    entries = [path / name for name in os.listdir(path)]
    body += render_directory_html("/" + path.relative_to(root).as_posix() + "/", entries).decode("utf-8").split("<main>", 1)[1].split("</main>", 1)[0]
    return page_shell(f"Event - {event['name']}", body)


def render_graph_gallery(root: Path, path: Path, url_path: str) -> bytes:
    images = scan_graph_images(root, path)
    title_path = unquote(url_path).strip("/") or "graphs"
    products = sorted({image["product"] for image in images})
    product_options = "".join(
        f'<option value="{html.escape(product)}">{html.escape(product)}</option>'
        for product in products
    )
    breadcrumbs = " / ".join(
        f'<a href="{href}">{html.escape(label)}</a>'
        for label, href in breadcrumb_items(url_path)
    )
    image_json = json.dumps(images, ensure_ascii=False)
    first = images[0] if images else None
    first_url = first["url"] if first else ""
    first_name = first["name"] if first else "No graphs"
    first_meta = f"{first['product']} / {first['time']}" if first else ""
    body = f"""
    <header>
      <div>
        <h1>{html.escape(title_path)}</h1>
        <div class="breadcrumbs">{breadcrumbs} / <a href="?view=list">file list</a></div>
      </div>
      <div class="stats">
        <div class="stat"><strong>{len(images)}</strong><span>graphs</span></div>
        <div class="stat"><strong>{len(products)}</strong><span>products</span></div>
      </div>
    </header>
    <div class="gallery-toolbar">
      <input id="graphSearch" placeholder="Search time or filename">
      <select id="productFilter"><option value="">All products</option>{product_options}</select>
      <button class="button" id="prevGraph" type="button">Prev</button>
      <button class="button" id="nextGraph" type="button">Next</button>
      <a class="button" id="openGraph" href="{html.escape(first_url)}">Open image</a>
    </div>
    <section class="graph-viewer">
      <div class="stage">
        <img id="mainGraph" src="{html.escape(first_url)}" alt="">
      </div>
      <aside>
        <div class="selected-meta">
          <strong id="selectedName">{html.escape(first_name)}</strong>
          <span id="selectedMeta">{html.escape(first_meta)}</span>
        </div>
        <div class="thumb-grid" id="thumbGrid"></div>
      </aside>
    </section>
    <script>
      const graphImages = {image_json};
      const searchInput = document.getElementById('graphSearch');
      const productFilter = document.getElementById('productFilter');
      const thumbGrid = document.getElementById('thumbGrid');
      const mainGraph = document.getElementById('mainGraph');
      const selectedName = document.getElementById('selectedName');
      const selectedMeta = document.getElementById('selectedMeta');
      const openGraph = document.getElementById('openGraph');
      let filtered = [...graphImages];
      let selectedIndex = 0;

      function selectGraph(index) {{
        if (!filtered.length) {{
          mainGraph.removeAttribute('src');
          selectedName.textContent = 'No matching graphs';
          selectedMeta.textContent = '';
          openGraph.removeAttribute('href');
          return;
        }}
        selectedIndex = (index + filtered.length) % filtered.length;
        const image = filtered[selectedIndex];
        mainGraph.src = image.url;
        selectedName.textContent = image.name;
        selectedMeta.textContent = `${{image.product}} / ${{image.time}} / ${{image.size}}`;
        openGraph.href = image.url;
        thumbGrid.querySelectorAll('button').forEach((button, idx) => {{
          button.classList.toggle('active', idx === selectedIndex);
        }});
      }}

      function renderThumbs() {{
        thumbGrid.replaceChildren(...filtered.map((image, idx) => {{
          const button = document.createElement('button');
          button.type = 'button';
          button.className = 'graph-thumb';
          button.title = image.name;
          button.innerHTML = `<strong>${{image.time}}</strong><span>${{image.product}}</span><small>${{image.name}}</small>`;
          button.addEventListener('click', () => selectGraph(idx));
          return button;
        }}));
        selectGraph(Math.min(selectedIndex, filtered.length - 1));
      }}

      function applyGraphFilters() {{
        const query = searchInput.value.toLowerCase();
        const product = productFilter.value;
        filtered = graphImages.filter(image => {{
          const text = `${{image.name}} ${{image.time}} ${{image.product}}`.toLowerCase();
          return (!query || text.includes(query)) && (!product || image.product === product);
        }});
        selectedIndex = 0;
        renderThumbs();
      }}

      document.getElementById('prevGraph').addEventListener('click', () => selectGraph(selectedIndex - 1));
      document.getElementById('nextGraph').addEventListener('click', () => selectGraph(selectedIndex + 1));
      searchInput.addEventListener('input', applyGraphFilters);
      productFilter.addEventListener('input', applyGraphFilters);
      window.addEventListener('keydown', event => {{
        if (event.key === 'ArrowLeft') selectGraph(selectedIndex - 1);
        if (event.key === 'ArrowRight') selectGraph(selectedIndex + 1);
      }});
      renderThumbs();
    </script>
    """
    extra_head = """
    <style>
      .gallery-toolbar { display: grid; grid-template-columns: minmax(180px, 1fr) 170px auto auto auto; gap: 10px; margin: 18px 0; align-items: center; }
      .gallery-toolbar input, .gallery-toolbar select { width: 100%; padding: 9px 10px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); color: var(--text); }
      .graph-viewer { display: grid; grid-template-columns: minmax(0, 1fr) 320px; gap: 12px; align-items: start; }
      .stage { min-height: 520px; display: grid; place-items: center; background: var(--surface); border: 1px solid var(--line); border-radius: 8px; overflow: hidden; box-shadow: var(--shadow); }
      .stage img { display: block; width: 100%; height: 100%; max-height: calc(100vh - 220px); object-fit: contain; background: #fff; }
      .graph-viewer aside { background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 10px; box-shadow: 0 4px 16px rgba(24,32,42,.04); }
      .selected-meta { display: grid; gap: 3px; padding: 4px 4px 10px; }
      .selected-meta strong { overflow-wrap: anywhere; }
      .selected-meta span { color: var(--muted); font-size: 12px; }
      .thumb-grid { display: grid; grid-template-columns: 1fr; gap: 6px; max-height: calc(100vh - 280px); overflow: auto; padding-right: 2px; }
      .graph-thumb { display: grid; grid-template-columns: 74px minmax(74px, auto) minmax(0, 1fr); gap: 8px; align-items: center; padding: 7px 8px; border: 1px solid var(--line); border-radius: 8px; background: #fff; color: var(--text); cursor: pointer; text-align: left; }
      .graph-thumb.active { border-color: var(--accent); box-shadow: 0 0 0 2px var(--accent-soft); }
      .graph-thumb strong { font-size: 13px; font-weight: 700; }
      .graph-thumb span { min-width: 0; padding: 2px 6px; border-radius: 999px; background: var(--soft); color: #475569; font-size: 12px; font-weight: 600; text-align: center; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      .graph-thumb small { min-width: 0; color: var(--muted); font-size: 12px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
      @media (max-width: 980px) {
        .gallery-toolbar { grid-template-columns: 1fr 1fr; }
        .graph-viewer { grid-template-columns: 1fr; }
        .stage { min-height: 360px; }
        .thumb-grid { max-height: 360px; }
      }
      @media (max-width: 620px) {
        .gallery-toolbar { grid-template-columns: 1fr; }
        .graph-thumb { grid-template-columns: 72px minmax(64px, auto); }
        .graph-thumb small { grid-column: 1 / -1; }
      }
    </style>
    """
    return page_shell(f"Graphs - {title_path}", body, extra_head=extra_head)


class PrettyDirectoryHandler(SimpleHTTPRequestHandler):
    server_version = "GNSSResultsHTTP/1.0"

    def _root_dir(self) -> Path:
        return Path(self.directory).resolve()

    def _safe_path_from_url(self, url_path: str) -> Path | None:
        root = self._root_dir()
        rel = unquote(urlparse(url_path).path).strip("/")
        candidate = (root / rel).resolve()
        if candidate == root or root in candidate.parents:
            return candidate
        return None

    def _send_bytes(self, payload: bytes, content_type: str = "text/html; charset=utf-8") -> None:
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_json(self, payload: object) -> None:
        encoded = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._send_bytes(encoded, "application/json; charset=utf-8")

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            return self.handle_api(parsed.path, parse_qs(parsed.query))

        root = self._root_dir()
        if parsed.path in {"", "/"}:
            return self._send_bytes(render_dashboard(root))

        candidate = self._safe_path_from_url(parsed.path)
        if (
            candidate
            and candidate.is_dir()
            and parse_qs(parsed.query).get("view") != ["list"]
            and is_graphs_dir(root, candidate)
        ):
            return self._send_bytes(render_graph_gallery(root, candidate, parsed.path))

        if candidate and candidate.is_dir() and is_event_dir(candidate):
            return self._send_bytes(render_event_page(root, candidate))

        return super().do_GET()

    def handle_api(self, path: str, query: dict[str, list[str]]):
        root = self._root_dir()
        events = scan_events(root)
        if path == "/api/summary":
            return self._send_json(build_summary(events))
        if path == "/api/events":
            return self._send_json(events)
        if path.startswith("/api/event/"):
            rel = unquote(path.removeprefix("/api/event/")).strip("/")
            for event in events:
                if event["path"] == rel or event["name"] == rel:
                    return self._send_json(event)
            self.send_error(404, "Event not found")
            return None
        self.send_error(404, "API endpoint not found")
        return None

    def list_directory(self, path):
        try:
            entries = [Path(path) / name for name in os.listdir(path)]
        except OSError:
            self.send_error(404, "No permission to list directory")
            return None

        encoded = render_directory_html(self.path, entries)
        response = BytesIO(encoded)
        response.seek(0)
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        return response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pretty directory listing for GNSS results.")
    parser.add_argument("--directory", default="./results", help="Directory to serve.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument("--bind", default="127.0.0.1", help="Address to bind.")
    args = parser.parse_args(argv)

    directory = Path(args.directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)

    handler = partial(PrettyDirectoryHandler, directory=str(directory))
    server = ThreadingHTTPServer((args.bind, args.port), handler)
    print(f"Serving {directory} at http://{args.bind}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
