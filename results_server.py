from __future__ import annotations

import argparse
import html
import os
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import quote, unquote


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


def file_kind(path: Path) -> str:
    if path.is_dir():
        return "Folder"
    return FILE_TYPE_LABELS.get(path.suffix.lower(), path.suffix[1:].upper() or "File")


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


class PrettyDirectoryHandler(SimpleHTTPRequestHandler):
    server_version = "GNSSResultsHTTP/1.0"

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
