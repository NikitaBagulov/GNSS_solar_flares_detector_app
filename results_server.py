from __future__ import annotations

import argparse
import html
import json
import os
import re
from datetime import datetime
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

from plot_requests import TTL_SECONDS, cleanup_cache, event_channels, event_series, map_epochs, normalize_email, plots_for_email, render_plot, store_plot
from plot_editor import editor_content


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
UI_TRANSLATIONS = {
    "Event catalog": "Каталог вспышек",
    "Browse events and see at a glance which data products are ready.": "Просматривайте вспышки и сразу проверяйте готовность результатов.",
    "Events JSON": "События JSON",
    "Summary JSON": "Сводка JSON",
    "Total events": "Всего событий",
    "Fully processed": "Обработано полностью",
    "Need attention": "Требуют внимания",
    "Catalog storage": "Размер каталога",
    "All events": "Все события",
    "Select an event to inspect its files and results.": "Откройте событие, чтобы посмотреть файлы и результаты.",
    "Search by date, event name, or class": "Поиск по дате, имени события или классу",
    "All classes": "Все классы",
    "Any status": "Любой статус",
    "Needs attention": "Требует внимания",
    "Newest first": "Сначала новые",
    "Oldest first": "Сначала старые",
    "Event name": "По имени события",
    "Largest first": "Сначала крупные",
    "Clear filters": "Сбросить фильтры",
    "Event / date": "Событие / дата",
    "Class": "Класс",
    "Processing progress": "Готовность обработки",
    "Source data": "Исходные данные",
    "Last updated": "Обновлено",
    "No events match these filters. Try a different search or clear the filters.": "События не найдены. Измените условия поиска или сбросьте фильтры.",
    "Date unavailable": "Дата не указана",
    "available": "есть",
    "missing": "нет",
    "Results": "Результаты",
    "GNSS SOLAR FLARE DETECTOR": "GNSS — ДЕТЕКТОР СОЛНЕЧНЫХ ВСПЫШЕК",
    "Event overview": "Обзор события",
    "Latest available event plot.": "Последний доступный график события.",
    "Availability of each calculated product.": "Наличие каждого рассчитанного продукта.",
    "Back to catalog": "Вернуться в каталог",
    "Event status": "Состояние события",
    "Products": "Продукты",
    "Combined": "Общие",
    "GOES X-ray": "Рентгеновские данные GOES",
    "SOHO SEM": "Данные SOHO SEM",
    "Processing status": "Состояние обработки",
    "Product": "Продукт",
    "Map": "Карта",
    "Index": "Индекс",
    "Ready": "Готово",
    "Missing": "Отсутствует",
    "Source measurements": "Исходные измерения",
    "Open results": "Перейти к результатам",
    "Maps and indices": "Карты и индексы",
    "Maps": "Карты",
    "Indices": "Индексы",
    "Graphs": "Графики",
    "Combined plots": "Общие графики",
    "Event files": "Файлы события",
    "Browse files": "Просмотреть файлы",
    "Preview": "Предпросмотр",
    "Event details": "Данные события",
    "Solar flare class": "Класс вспышки",
    "Event date": "Дата события",
    "Data size": "Объём данных",
    "No preview graph is available yet.": "График для предпросмотра пока не создан.",
    "yes": "да",
    "no": "нет",
    "present": "есть",
    "Maps ready": "Карты готовы",
    "Indices ready": "Индексы готовы",
    "graphs": "графиков",
    "maps": "карт",
    "indices": "индексов",
    "size": "объём",
    "Date": "Дата",
    "Files": "Файлы",
    "Name": "Имя",
    "Type": "Тип",
    "Modified": "Изменён",
    "Size": "Размер",
    "Folder": "Папка",
    "File": "Файл",
    "Parent": "Вверх",
    "folders": "папок",
    "files": "файлов",
    "file size": "размер файлов",
    "No files in this folder.": "В этой папке нет файлов.",
    "Prev": "Назад",
    "Next": "Далее",
    "Open image": "Открыть изображение",
    "Search time or filename": "Поиск по времени или имени файла",
    "All products": "Все продукты",
    "products": "продуктов",
    "No graphs": "Нет графиков",
    "No matching graphs": "Подходящих графиков нет",
    "file list": "список файлов",
    "Language": "Язык",
    "Create a plot": "Создать график",
    "Open plot studio": "Открыть редактор графиков",
    "Plot studio": "Редактор графиков",
    "Back to event": "Вернуться к событию",
    "Layout preview": "Предпросмотр расположения",
    "Figure template": "Шаблон фигуры",
    "Flare overview · map + Sun + flux": "Обзор вспышки · карта + Солнце + потоки",
    "Solar irradiance · Sun + GOES + SOHO": "Солнечное излучение · Солнце + GOES + SOHO",
    "Ionospheric response · map + index + flux": "Отклик ионосферы · карта + индекс + поток",
    "Map comparison · two products": "Сравнение карт · два продукта",
    "Time series · stacked panels": "Временные ряды · друг под другом",
    "Custom layout": "Своя компоновка",
    "Choose a template, then drag, resize or edit panels to customize it.": "Выберите шаблон, затем перетаскивайте, меняйте размеры и содержимое панелей.",
    "Drag, resize or edit any panel to make this template your own.": "Перетащите, измените размер или данные любой панели, чтобы настроить шаблон под себя.",
    "Custom layout · your changes are kept until you choose another template.": "Своя компоновка · изменения сохраняются, пока вы не выберете другой шаблон.",
    "Panel layout, event markers and observation time are previewed here; actual data appears after generation.": "Здесь показаны расположение панелей, метки события и время наблюдения; данные появятся после построения.",
    "Selected panel": "Выбранная панель",
    "Plot style": "Стиль графика",
    "Simple · white, fine grid": "Простой · белый фон, тонкая сетка",
    "Plotter · colored axes": "Plotter · цветные оси",
    "Width (%)": "Ширина (%)",
    "Height (%)": "Высота (%)",
    "Observation time (UTC)": "Время наблюдения (UTC)",
    "One time for all maps and time-series panels, from the available HDF5 datasets.": "Одно время для всех карт и временных рядов, из доступных данных HDF5.",
    "Fixed map scales: ROTI · viridis 0–0.5 TECu/min; dTEC · RdBu_r −0.5…0.5 TECu.": "Постоянные шкалы карт: ROTI · viridis 0–0.5 TECu/мин; dTEC · RdBu_r −0.5…0.5 TECu.",
    "Line color": "Цвет линии",
    "Remove selected panel": "Удалить панель",
    "Generate plot": "Построить график",
    "Open full-size plot": "Открыть график в полном размере",
    "Use the same email in the catalog to find your plots. No messages are sent.": "Введите ту же почту в каталоге, чтобы найти графики. Письма не отправляются.",
    "Drag panels by their headers and resize from the lower-right corner.": "Перетаскивайте панели за заголовки и меняйте размер за правый нижний угол.",
    "Build plot": "Построить график",
    "Add panel": "Добавить панель",
    "Selected plots": "Выбранные графики",
    "Layout": "Расположение",
    "Vertical": "Друг под другом",
    "Two columns": "Два столбца",
    "Plot title": "Название графика",
    "Additional caption": "Дополнительная подпись",
    "Optional description": "Необязательное описание",
    "Solar disk": "Солнечный диск",
    "Flare position unavailable": "Положение вспышки неизвестно",
    "Email identifier": "Почта для поиска графиков",
    "My plots": "Мои графики",
    "Enter your email to find plots created with it during the last three days. No messages are sent.": "Укажите почту, чтобы найти созданные с ней графики за последние три дня. Письма не отправляются.",
    "Find my plots": "Найти мои графики",
    "Enter your email address to save and find this plot. No messages are sent.": "Укажите почту, чтобы сохранить и найти график. Письма не отправляются.",
    "Generate and open": "Построить и открыть",
    "Remove": "Удалить",
    "Color": "Цвет",
    "Data series": "Ряд данных",
    "Plots are generated on request and available for three days.": "Графики строятся по запросу и доступны три дня.",
    "No data series available for this event.": "Для этого события нет данных для графиков.",
    "Generating plot…": "Построение графика…",
}

LANGUAGE_SCRIPT = r"""
(() => {
  const picker = document.getElementById('languageSelect');
  if (!picker) return;
  const translations = JSON.parse(document.getElementById('uiTranslations').textContent);
  const reverse = Object.fromEntries(Object.entries(translations).map(([en, ru]) => [ru, en]));
  function setLanguage(language) {
    const dictionary = language === 'ru' ? translations : reverse;
    const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    const textNodes = [];
    while (walker.nextNode()) textNodes.push(walker.currentNode);
    for (const node of textNodes) {
      const original = node.nodeValue;
      const trimmed = original.trim();
      let replacement = dictionary[trimmed];
      let match = trimmed.match(/^(Maps|Indices|Graphs|Карты|Индексы|Графики) (\d+(?:\/\d+)?)$/);
      if (language === 'ru' && match) replacement = `${({Maps: 'Карты', Indices: 'Индексы', Graphs: 'Графики', Карты: 'Карты', Индексы: 'Индексы', Графики: 'Графики'})[match[1]]} ${match[2]}`;
      if (language === 'en' && match) replacement = `${({Maps: 'Maps', Indices: 'Indices', Graphs: 'Graphs', Карты: 'Maps', Индексы: 'Indices', Графики: 'Graphs'})[match[1]]} ${match[2]}`;
      match = trimmed.match(/^Class ([A-Z?])$/);
      if (language === 'ru' && match) replacement = `Класс ${match[1]}`;
      match = trimmed.match(/^Класс ([A-Z?])$/);
      if (language === 'en' && match) replacement = `Class ${match[1]}`;
      match = trimmed.match(/^(\d+) of (\d+) events$/);
      if (language === 'ru' && match) replacement = `${match[1]} из ${match[2]} событий`;
      match = trimmed.match(/^(\d+) из (\d+) событий$/);
      if (language === 'en' && match) replacement = `${match[1]} of ${match[2]} events`;
      match = trimmed.match(/^\((\d+) items\)$/);
      if (language === 'ru' && match) replacement = `(объектов: ${match[1]})`;
      match = trimmed.match(/^\(объектов: (\d+)\)$/);
      if (language === 'en' && match) replacement = `(${match[1]} items)`;
      if (replacement) node.nodeValue = original.replace(trimmed, replacement);
    }
    for (const element of document.querySelectorAll('[placeholder], [title], [aria-label]')) {
      for (const attribute of ['placeholder', 'title', 'aria-label']) {
        const value = element.getAttribute(attribute);
        if (value && dictionary[value]) element.setAttribute(attribute, dictionary[value]);
      }
    }
    document.documentElement.lang = language;
    document.getElementById('languageLabel').textContent = language === 'ru' ? 'Язык' : 'Language';
    picker.value = language;
    try { localStorage.setItem('gnss-results-language', language); } catch (_) {}
  }
  picker.addEventListener('change', () => setLanguage(picker.value));
  let saved = 'en';
  try { saved = localStorage.getItem('gnss-results-language') || 'en'; } catch (_) {}
  setLanguage(saved === 'ru' ? 'ru' : 'en');
})();
"""


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
    translations_json = json.dumps(UI_TRANSLATIONS, ensure_ascii=False).replace("<", "\\u003c")
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
    .app-topbar {{ width: min(1180px, calc(100vw - 32px)); min-height: 52px; display: flex; justify-content: space-between; align-items: center; margin: 0 auto; border-bottom: 1px solid var(--line); }}
    .app-brand {{ color: var(--text); font-size: 13px; font-weight: 700; text-decoration: none; }}
    .language-control {{ display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 12px; }}
    .language-control select {{ min-height: 32px; padding: 4px 28px 4px 9px; border: 1px solid var(--line); border-radius: 7px; background: var(--surface); color: var(--text); font: inherit; cursor: pointer; }}
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
  <div class="app-topbar">
    <a class="app-brand" href="/">GNSS · Solar Flare</a>
    <label class="language-control" for="languageSelect"><span id="languageLabel">Language</span>
      <select id="languageSelect" aria-label="Language"><option value="en">English</option><option value="ru">Русский</option></select>
    </label>
  </div>
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
  <script type="application/json" id="uiTranslations">{translations_json}</script>
  <script>{LANGUAGE_SCRIPT}</script>
</body>
</html>
"""
    return html_doc.encode("utf-8", "surrogateescape")


def page_shell(title: str, body: str, extra_head: str = "") -> bytes:
    translations_json = json.dumps(UI_TRANSLATIONS, ensure_ascii=False).replace("<", "\\u003c")
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
    .app-topbar {{ width: min(1320px, calc(100vw - 32px)); min-height: 52px; display: flex; justify-content: space-between; align-items: center; margin: 0 auto; border-bottom: 1px solid var(--line); }}
    .app-brand {{ color: var(--text); font-size: 13px; font-weight: 700; text-decoration: none; }}
    .language-control {{ display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 12px; }}
    .language-control select {{ min-height: 32px; padding: 4px 28px 4px 9px; border: 1px solid var(--line); border-radius: 7px; background: var(--surface); color: var(--text); font: inherit; cursor: pointer; }}
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
      .app-topbar {{ width: min(100vw - 20px, 1320px); }}
      header {{ display: block; }}
      h1 {{ font-size: 22px; }}
      .stats {{ justify-content: flex-start; margin-top: 14px; }}
      .toolbar {{ grid-template-columns: 1fr; }}
    }}
  </style>
  {extra_head}
</head>
<body>
  <div class="app-topbar">
    <a class="app-brand" href="/">GNSS · Solar Flare</a>
    <label class="language-control" for="languageSelect"><span id="languageLabel">Language</span>
      <select id="languageSelect" aria-label="Language"><option value="en">English</option><option value="ru">Русский</option></select>
    </label>
  </div>
  <main>{body}</main>
  <script type="application/json" id="uiTranslations">{translations_json}</script>
  <script>{LANGUAGE_SCRIPT}</script>
</body>
</html>
"""
    return html_doc.encode("utf-8", "surrogateescape")


def render_dashboard(root: Path) -> bytes:
    events = scan_events(root)
    summary = build_summary(events)
    class_order = {"X": 0, "M": 1, "C": 2, "B": 3, "A": 4}
    classes = sorted({event["class"] for event in events}, key=lambda value: (class_order.get(value, 5), value))
    class_options = "".join(
        f'<option value="{html.escape(klass)}">Class {html.escape(klass)}</option>'
        for klass in classes
    )
    rows = []
    for event in events:
        rows.append(
            f"""<tr data-name="{html.escape(event['name'].lower())}" data-date="{html.escape(event['date'])}" data-class="{html.escape(event['class'])}" data-complete="{str(event['complete']).lower()}" data-size="{event['size_bytes']}">
              <td class="event-cell"><a class="name-link" href="{html.escape(event['url'])}">{html.escape(event['name'])}</a><span>{html.escape(event['date'] or 'Date unavailable')}</span></td>
              <td><span class="class-tag class-{html.escape(event['class'].lower())}">{html.escape(event['class'])}</span></td>
              <td><div class="progress-list">
                <span class="badge {'ok' if event['maps_ready'] == event['maps_total'] else 'warn'}">Maps {event['maps_ready']}/{event['maps_total']}</span>
                <span class="badge {'ok' if event['indices_ready'] == event['indices_total'] else 'warn'}">Indices {event['indices_ready']}/{event['indices_total']}</span>
                <span class="badge {'ok' if event['graphs_count'] else 'bad'}">Graphs {event['graphs_count']}</span>
              </div></td>
              <td><div class="progress-list source-list">
                <span class="source {'ok' if event['sources'].get('goes_xray') else 'missing'}" title="GOES X-ray data">GOES {'available' if event['sources'].get('goes_xray') else 'missing'}</span>
                <span class="source {'ok' if event['sources'].get('soho_sem') else 'missing'}" title="SOHO SEM data">SOHO {'available' if event['sources'].get('soho_sem') else 'missing'}</span>
              </div></td>
              <td class="muted">{html.escape(event['modified'])}</td>
            </tr>"""
        )

    body = f"""
    <header>
      <div>
        <div class="eyebrow">GNSS SOLAR FLARE DETECTOR</div>
        <h1>Event catalog</h1>
        <p class="page-intro">Browse events and see at a glance which data products are ready.</p>
      </div>
      <nav class="api-links" aria-label="Data endpoints">
        <a href="/api/events">Events JSON</a><a href="/api/summary">Summary JSON</a>
      </nav>
    </header>
    <section class="catalog-summary" aria-label="Catalog overview">
      <div class="summary-card"><span class="summary-icon">▦</span><div><strong>{summary['events']}</strong><span>Total events</span></div></div>
      <div class="summary-card"><span class="summary-icon ready-icon">✓</span><div><strong>{summary['complete']}</strong><span>Fully processed</span></div></div>
      <div class="summary-card"><span class="summary-icon attention-icon">!</span><div><strong>{summary['incomplete']}</strong><span>Need attention</span></div></div>
      <div class="catalog-storage"><span>Catalog storage</span><strong>{html.escape(summary['size'])}</strong></div>
    </section>
    <section class="catalog-panel my-plots" aria-label="My plots">
      <div class="catalog-heading"><div><h2>My plots</h2><span class="muted">Enter your email to find plots created with it during the last three days. No messages are sent.</span></div></div>
      <form id="myPlotsForm" class="my-plots-form"><label>Email identifier <input id="myPlotsEmail" type="email" required autocomplete="email" placeholder="name@example.com"></label><button class="button primary-button" type="submit">Find my plots</button></form>
      <p id="myPlotsStatus" class="muted" role="status" aria-live="polite"></p><div id="myPlotsResults" class="my-plots-results"></div>
    </section>
    <section class="catalog-panel" aria-label="Solar flare events">
      <div class="catalog-heading">
        <div><h2>All events</h2><span class="muted">Select an event to inspect its files and results.</span></div>
        <span id="resultsCount" class="result-count" aria-live="polite">{summary['events']} events</span>
      </div>
      <div class="toolbar" role="search">
        <label class="search-field"><span class="sr-only">Search events</span><span aria-hidden="true">⌕</span><input id="q" type="search" placeholder="Search by date, event name, or class" autocomplete="off"></label>
        <label><span class="sr-only">Filter by class</span><select id="classFilter"><option value="">All classes</option>{class_options}</select></label>
        <label><span class="sr-only">Filter by processing status</span><select id="statusFilter"><option value="">Any status</option><option value="complete">Fully processed</option><option value="incomplete">Needs attention</option></select></label>
        <label><span class="sr-only">Sort events</span><select id="sortBy"><option value="newest">Newest first</option><option value="oldest">Oldest first</option><option value="name">Event name</option><option value="size">Largest first</option></select></label>
        <button class="reset-button" id="resetFilters" type="button">Clear filters</button>
      </div>
      <div class="table-wrap">
        <table id="eventsTable">
          <thead><tr><th>Event / date</th><th>Class</th><th>Processing progress</th><th>Source data</th><th>Last updated</th></tr></thead>
          <tbody>{''.join(rows)}<tr id="noResults" hidden><td colspan="5" class="empty-state">No events match these filters. Try a different search or clear the filters.</td></tr></tbody>
        </table>
      </div>
    </section>
    <script>
      document.getElementById('myPlotsForm').addEventListener('submit', async event => {{
        event.preventDefault();
        const results = document.getElementById('myPlotsResults');
        const status = document.getElementById('myPlotsStatus');
        const text = (en, ru) => document.documentElement.lang === 'ru' ? ru : en;
        results.replaceChildren();
        status.textContent = text('Searching…', 'Поиск…');
        try {{
          const response = await fetch('/api/plots/search', {{method: 'POST', headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{email: document.getElementById('myPlotsEmail').value}})}});
          const payload = await response.json();
          if (!response.ok) throw Error(payload.error);
          status.textContent = payload.plots.length
            ? text(`${{payload.plots.length}} plots found`, `Найдено графиков: ${{payload.plots.length}}`)
            : text('No saved plots for this email.', 'Для этой почты сохранённых графиков нет.');
          for (const plot of payload.plots) {{
            const row = document.createElement('div'); row.className = 'my-plot-row';
            const link = document.createElement('a'); link.href = plot.url; link.target = '_blank'; link.rel = 'noopener';
            link.textContent = plot.title || plot.event;
            const description = document.createElement('span'); description.textContent = `${{plot.event}} · ${{plot.created_at}}`;
            const remove = document.createElement('button'); remove.type = 'button'; remove.className = 'button';
            remove.textContent = text('Delete plot', 'Удалить график');
            remove.onclick = async () => {{
              const deleted = await fetch(plot.delete_url, {{method: 'DELETE'}});
              if (deleted.ok) {{ row.remove(); status.textContent = text('Plot deleted.', 'График удалён.'); }}
            }};
            row.append(link, description, remove); results.append(row);
          }}
        }} catch (error) {{ status.textContent = error.message; }}
      }});
      const q = document.getElementById('q');
      const classFilter = document.getElementById('classFilter');
      const statusFilter = document.getElementById('statusFilter');
      const sortBy = document.getElementById('sortBy');
      const tbody = document.querySelector('#eventsTable tbody');
      const originalRows = Array.from(tbody.querySelectorAll('tr[data-name]'));
      const noResults = document.getElementById('noResults');
      const resultsCount = document.getElementById('resultsCount');
      function applyFilters() {{
        const query = q.value.toLowerCase();
        const klass = classFilter.value;
        const status = statusFilter.value;
        let rows = originalRows.filter(row => row !== noResults && (() => {{
          const text = (row.dataset.name + ' ' + row.dataset.date + ' ' + row.dataset.class).toLowerCase();
          const okQuery = !query || text.includes(query);
          const okClass = !klass || row.dataset.class === klass;
          const okStatus = !status || (status === 'complete') === (row.dataset.complete === 'true');
          return okQuery && okClass && okStatus;
        }})());
        rows.sort((a, b) => {{
          if (sortBy.value === 'size') return Number(b.dataset.size) - Number(a.dataset.size);
          if (sortBy.value === 'name') return a.dataset.name.localeCompare(b.dataset.name);
          if (!a.dataset.date && b.dataset.date) return 1;
          if (a.dataset.date && !b.dataset.date) return -1;
          const order = a.dataset.date.localeCompare(b.dataset.date);
          return sortBy.value === 'oldest' ? order : -order;
        }});
        tbody.replaceChildren(...rows);
        noResults.hidden = rows.length !== 0;
        tbody.append(noResults);
        resultsCount.textContent = document.documentElement.lang === 'ru'
          ? `${{rows.length}} из ${{originalRows.length}} событий`
          : `${{rows.length}} of ${{originalRows.length}} events`;
      }}
      [q, classFilter, statusFilter, sortBy].forEach(el => el.addEventListener('input', applyFilters));
      document.getElementById('resetFilters').addEventListener('click', () => {{
        q.value = ''; classFilter.value = ''; statusFilter.value = ''; sortBy.value = 'newest'; applyFilters(); q.focus();
      }});
      applyFilters();
    </script>
    """
    extra_head = """
    <style>
      main { width: min(1380px, calc(100vw - 48px)); padding-top: 38px; }
      header { align-items: center; }
      .eyebrow { color: var(--accent); font-size: 11px; font-weight: 800; letter-spacing: .12em; margin-bottom: 8px; }
      h1 { font-size: clamp(28px, 3vw, 38px); letter-spacing: -.03em; margin-bottom: 5px; }
      .page-intro { margin: 0; color: var(--muted); font-size: 15px; }
      .api-links { display: flex; gap: 8px; }
      .api-links a { color: var(--muted); font-size: 12px; text-decoration: none; padding: 7px 10px; border: 1px solid var(--line); border-radius: 7px; background: var(--surface); }
      .api-links a:hover { color: var(--accent); border-color: var(--accent); }
      .catalog-summary { display: grid; grid-template-columns: repeat(3, minmax(150px, 1fr)) minmax(140px, .8fr); gap: 12px; margin: 24px 0 18px; }
      .my-plots { margin-bottom: 18px; }
      .my-plots-form { display: flex; align-items: end; flex-wrap: wrap; gap: 10px; margin: 10px 20px 16px; }
      .my-plots-form label { display: grid; gap: 5px; flex: 1 1 250px; max-width: 440px; }
      .my-plots-form input { min-height: 40px; padding: 9px 11px; border: 1px solid var(--line); border-radius: 8px; font: inherit; }
      #myPlotsStatus { margin: 0 20px 12px; }
      #myPlotsStatus:empty { display: none; }
      .my-plots-results { margin: 0 20px 16px; }
      .my-plot-row { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; padding: 10px 0; border-top: 1px solid var(--line); }
      .my-plot-row span { color: var(--muted); font-size: 12px; flex: 1; }
      .summary-card, .catalog-storage { min-height: 86px; display: flex; align-items: center; gap: 12px; padding: 16px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
      .summary-card > div, .catalog-storage { display: flex; flex-direction: column; }
      .summary-card strong, .catalog-storage strong { font-size: 22px; line-height: 1.15; }
      .summary-card div span, .catalog-storage span { color: var(--muted); font-size: 12px; }
      .summary-icon { width: 34px; height: 34px; display: grid; place-items: center; border-radius: 9px; background: #e8eef5; color: #475569; font-weight: 800; }
      .ready-icon { background: #dff3ef; color: var(--good); }
      .attention-icon { background: #fff3d6; color: var(--warn); }
      .catalog-storage { justify-content: center; gap: 3px; }
      .catalog-panel { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); overflow: hidden; }
      .catalog-heading { display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 20px 20px 8px; }
      .catalog-heading h2 { margin: 0 0 3px; font-size: 18px; }
      .result-count { color: var(--muted); font-size: 13px; white-space: nowrap; }
      .toolbar { display: grid; grid-template-columns: minmax(220px, 1fr) 150px 190px 150px auto; gap: 9px; margin: 12px 20px 16px; }
      .toolbar input, .toolbar select { width: 100%; min-height: 40px; padding: 9px 11px; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); color: var(--text); font: inherit; }
      .search-field { position: relative; }
      .search-field > span:not(.sr-only) { position: absolute; left: 11px; top: 5px; color: var(--muted); font-size: 21px; }
      .search-field input { padding-left: 34px; }
      .reset-button { min-height: 40px; border: 0; background: transparent; color: var(--accent); font: inherit; cursor: pointer; white-space: nowrap; }
      .reset-button:hover { text-decoration: underline; }
      .sr-only { position: absolute; width: 1px; height: 1px; padding: 0; margin: -1px; overflow: hidden; clip: rect(0,0,0,0); white-space: nowrap; border: 0; }
      .table-wrap { border: 0; border-top: 1px solid var(--line); border-radius: 0; box-shadow: none; }
      table { min-width: 820px; }
      th, td { padding: 13px 18px; vertical-align: middle; }
      th { text-transform: none; letter-spacing: .01em; background: #fafbfc; }
      tbody tr:hover { background: #f7faf9; }
      .event-cell { min-width: 230px; }
      .event-cell .name-link { display: block; }
      .event-cell > span { color: var(--muted); font-size: 12px; }
      .class-tag { display: inline-grid; min-width: 34px; min-height: 28px; place-items: center; border-radius: 7px; background: #eef2f7; color: #475569; font-weight: 800; }
      .class-x { background: #fee4e2; color: #b42318; }
      .class-m { background: #fff0d9; color: #a15c00; }
      .class-c { background: #e9f2ff; color: #175cd3; }
      .progress-list { display: flex; flex-wrap: wrap; gap: 5px; }
      .progress-list .badge { min-width: 0; font-size: 11px; }
      .source { padding: 3px 7px; border-radius: 999px; font-size: 11px; font-weight: 700; }
      .source.ok { background: #dff3ef; color: var(--good); }
      .source.missing { background: #fee4e2; color: var(--bad); }
      .empty-state { padding: 36px 20px; text-align: center; color: var(--muted); white-space: normal; }
      @media (max-width: 900px) { .catalog-summary { grid-template-columns: repeat(2, minmax(140px, 1fr)); } .toolbar { grid-template-columns: repeat(2, minmax(0, 1fr)); } .search-field { grid-column: 1 / -1; } }
      @media (max-width: 560px) { main { width: calc(100vw - 24px); padding-top: 22px; } .catalog-summary { gap: 8px; } .summary-card, .catalog-storage { min-height: 72px; padding: 11px; } .catalog-heading { align-items: flex-start; padding: 16px 14px 6px; } .toolbar { margin: 10px 14px 14px; gap: 7px; } .api-links { margin-top: 14px; } header { margin-bottom: 12px; } }
    </style>
    """
    return page_shell("Event catalog · GNSS Solar Flare Detector", body, extra_head=extra_head)


def render_event_page(root: Path, path: Path) -> bytes:
    event = scan_event(root, path)
    available_series = event_series(path)
    product_rows = []
    for product in PRODUCTS:
        product_rows.append(
            f"""<tr>
              <td>{html.escape(product)}</td>
              <td><span class="badge {'ok' if event['maps'][product] else 'bad'}">{'Ready' if event['maps'][product] else 'Missing'}</span></td>
              <td><span class="badge {'ok' if event['indices'][product] else 'bad'}">{'Ready' if event['indices'][product] else 'Missing'}</span></td>
            </tr>"""
        )
    preview = event["preview_url"]
    preview_html = (
        f'<a href="{html.escape(preview)}"><img class="event-preview-image" src="{html.escape(preview)}" alt="Preview graph for {html.escape(event["name"])}"></a>'
        if preview
        else '<div class="event-preview-empty">No preview graph is available yet.</div>'
    )
    source_rows = []
    for key, label in (("goes_xray", "GOES X-ray"), ("soho_sem", "SOHO SEM")):
        available = event["sources"].get(key, False)
        source_rows.append(
            f'<div class="source-status"><span>{label}</span><span class="badge {"ok" if available else "bad"}">{"Ready" if available else "Missing"}</span></div>'
        )
    entries = [path / name for name in os.listdir(path)]
    directory_html = render_directory_html("/" + path.relative_to(root).as_posix() + "/", entries).decode("utf-8")
    files_table = directory_html.split('<div class="table-wrap">', 1)[1].split("</div>", 1)[0]
    complete_label = "Fully processed" if event["complete"] else "Needs attention"
    body = f"""
    <header class="event-hero">
      <div class="event-heading">
        <a class="back-link" href="/">← <span>Back to catalog</span></a>
        <div class="event-labels"><span class="class-tag class-{html.escape(event['class'].lower())}">{html.escape(event['class'])}</span><span class="badge {'ok' if event['complete'] else 'warn'}">{complete_label}</span></div>
        <h1>{html.escape(event['name'])}</h1>
        <p class="page-intro"><span>Event date</span>: {html.escape(event['date'] or 'Date unavailable')}</p>
      </div>
    </header>
    <section class="event-metrics" aria-label="Event overview">
      <div class="event-metric"><span class="metric-icon">▦</span><div><strong>{event['maps_ready']}/{event['maps_total']}</strong><span>Maps ready</span></div></div>
      <div class="event-metric"><span class="metric-icon">∑</span><div><strong>{event['indices_ready']}/{event['indices_total']}</strong><span>Indices ready</span></div></div>
      <div class="event-metric"><span class="metric-icon">◉</span><div><strong>{event['graphs_count']}</strong><span>graphs</span></div></div>
      <div class="event-metric"><span class="metric-icon">↗</span><div><strong>{html.escape(event['size'])}</strong><span>size</span></div></div>
    </section>
    <div class="event-main-grid">
      <section class="panel event-preview-panel">
        <div class="section-heading"><div><h2>Preview</h2><p class="muted">Latest available event plot.</p></div>{f'<a class="button" href="graphs/">Open results</a>' if event['graphs_count'] else ''}</div>
        {preview_html}
      </section>
      <aside class="panel event-details-panel">
        <h2>Event details</h2>
        <div class="detail-line"><span>Event date</span><strong>{html.escape(event['date'] or 'Date unavailable')}</strong></div>
        <div class="detail-line"><span>Solar flare class</span><strong>{html.escape(event['class'])}</strong></div>
        <div class="detail-line"><span>Data size</span><strong>{html.escape(event['size'])}</strong></div>
        <h3>Source measurements</h3>
        {''.join(source_rows)}
      </aside>
    </div>
    <section class="panel plot-editor" id="plotEditor">
      <div class="section-heading"><div><h2>Create a plot</h2><p class="muted">Plots are generated on request and available for three days.</p></div>
        {f'<a class="button primary-button" href="/editor/{quote(event["path"], safe="/")}" target="_blank" rel="noopener">Open plot studio</a>' if available_series else ''}</div>
      {'<p>No data series available for this event.</p>' if not available_series else ''}
    </section>
    <section class="panel processing-panel">
      <div class="section-heading"><div><h2>Processing status</h2><p class="muted">Availability of each calculated product.</p></div></div>
      <div class="table-wrap"><table><thead><tr><th>Product</th><th>Map</th><th>Index</th></tr></thead><tbody>{''.join(product_rows)}</tbody></table></div>
      <nav class="result-actions" aria-label="Open results">
        <span>Open results</span>
        {'<a class="button primary-button" href="maps/">Maps</a>' if (path / 'maps').is_dir() else ''}
        {'<a class="button" href="indices/">Indices</a>' if (path / 'indices').is_dir() else ''}
        {'<a class="button" href="graphs/">Graphs</a>' if (path / 'graphs').is_dir() else ''}
        {'<a class="button" href="graphs/combined/">Combined plots</a>' if (path / 'graphs' / 'combined').is_dir() else ''}
      </nav>
    </section>
    <details class="files-disclosure">
      <summary>Browse files <span class="muted">({len(entries)} items)</span></summary>
      <div class="table-wrap event-files-table">{files_table}</div>
    </details>
    """
    extra_head = """
    <style>
      main { width: min(1380px, calc(100vw - 48px)); padding-top: 28px; }
      .event-hero { align-items: flex-start; margin-bottom: 14px; }
      .event-heading { display: grid; gap: 9px; }
      .back-link { width: fit-content; color: var(--accent); font-weight: 600; text-decoration: none; }
      .back-link:hover { text-decoration: underline; }
      .event-labels { display: flex; align-items: center; gap: 8px; }
      .event-heading h1 { margin: 0; font-size: clamp(25px, 3vw, 34px); overflow-wrap: anywhere; }
      .event-heading .page-intro { margin: 0; }
      .class-tag { display: inline-grid; min-width: 34px; min-height: 28px; place-items: center; border-radius: 7px; background: #eef2f7; color: #475569; font-weight: 800; }
      .class-x { background: #fee4e2; color: #b42318; }
      .class-m { background: #fff0d9; color: #a15c00; }
      .class-c { background: #e9f2ff; color: #175cd3; }
      .event-metrics { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 18px 0; }
      .event-metric { display: flex; align-items: center; gap: 11px; min-height: 78px; padding: 13px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
      .metric-icon { display: grid; width: 34px; height: 34px; flex: 0 0 auto; place-items: center; border-radius: 9px; background: #e8eef5; color: var(--accent); font-size: 18px; font-weight: 700; }
      .event-metric div { display: grid; gap: 2px; }
      .event-metric strong { font-size: 18px; line-height: 1.1; }
      .event-metric div span { color: var(--muted); font-size: 12px; }
      .event-main-grid { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(260px, .8fr); gap: 12px; align-items: stretch; }
      .panel { padding: 18px; border-radius: 10px; }
      .event-main-grid h2, .processing-panel h2 { margin: 0; font-size: 17px; }
      .section-heading { display: flex; justify-content: space-between; align-items: center; gap: 12px; margin-bottom: 14px; }
      .section-heading p { margin: 3px 0 0; font-size: 12px; }
      .event-preview-image, .event-preview-empty { display: grid; width: 100%; height: min(48vw, 410px); min-height: 220px; place-items: center; object-fit: contain; background: #f7f9fb; border: 1px solid var(--line); border-radius: 7px; }
      .event-preview-empty { color: var(--muted); text-align: center; padding: 20px; }
      .event-details-panel h2 { margin-bottom: 8px; }
      .detail-line, .source-status { display: flex; justify-content: space-between; gap: 12px; align-items: center; padding: 10px 0; border-bottom: 1px solid var(--line); }
      .detail-line span, .source-status > span:first-child { color: var(--muted); }
      .detail-line strong { text-align: right; overflow-wrap: anywhere; }
      .event-details-panel h3 { margin: 22px 0 4px; font-size: 14px; }
      .source-status:last-child { border-bottom: 0; }
      .processing-panel { margin-top: 12px; }
      .plot-editor { margin-top: 12px; }
      .processing-panel .table-wrap { box-shadow: none; border-radius: 7px; }
      .processing-panel table { min-width: 480px; }
      .processing-panel th { text-transform: none; background: #fafbfc; }
      .result-actions { display: flex; align-items: center; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
      .result-actions > span { margin-right: 4px; color: var(--muted); font-weight: 600; }
      .primary-button { border-color: var(--accent); background: var(--accent); color: #fff; }
      .primary-button:hover { color: #fff; filter: brightness(.95); }
      .files-disclosure { margin-top: 12px; background: var(--surface); border: 1px solid var(--line); border-radius: 10px; }
      .files-disclosure summary { padding: 15px 18px; cursor: pointer; font-weight: 700; }
      .files-disclosure summary::marker { color: var(--accent); }
      .event-files-table { border: 0; border-top: 1px solid var(--line); border-radius: 0 0 10px 10px; box-shadow: none; }
      .event-files-table table { min-width: 560px; }
      @media (max-width: 820px) { .event-main-grid { grid-template-columns: 1fr; } .event-preview-image, .event-preview-empty { height: 52vw; } }
      @media (max-width: 600px) { main { width: calc(100vw - 24px); padding-top: 20px; } .event-metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 7px; } .event-metric { min-height: 68px; padding: 9px; gap: 8px; } .event-metric strong { font-size: 16px; } .event-metric div span { font-size: 11px; } .panel { padding: 14px; } .event-hero { display: block; } .result-actions { align-items: flex-start; } .result-actions > span { width: 100%; } }
    </style>
    """
    return page_shell(f"{event['name']} · GNSS Solar Flare", body, extra_head=extra_head)


def render_plot_editor_page(root: Path, path: Path) -> bytes:
    from flare_metadata import flare_metadata
    event = scan_event(root, path)
    epochs = map_epochs(path)
    series = {key: label for key, label in event_series(path).items() if not key.startswith("map:") or key in epochs}
    body, styles = editor_content(event, series, epochs, flare_metadata(path), event_channels(path))
    return page_shell(f"Plot studio · {event['name']}", body, extra_head=styles)


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
        if ".plot-cache" in unquote(parsed.path).split("/"):
            self.send_error(404)
            return
        if parsed.path.startswith("/generated/"):
            name = parsed.path.removeprefix("/generated/")
            if not re.fullmatch(r"[0-9a-f]{32}\.png", name):
                self.send_error(404)
                return
            cache = self._root_dir() / ".plot-cache"
            cleanup_cache(cache)
            image = cache / name
            if not image.is_file():
                self.send_error(404)
                return
            return self._send_bytes(image.read_bytes(), "image/png")
        if parsed.path.startswith("/api/"):
            return self.handle_api(parsed.path, parse_qs(parsed.query))

        root = self._root_dir()
        if parsed.path in {"", "/"}:
            return self._send_bytes(render_dashboard(root))

        if parsed.path.startswith("/editor/"):
            rel = unquote(parsed.path.removeprefix("/editor/")).strip("/")
            if rel not in {event["path"] for event in scan_events(root)}:
                self.send_error(404)
                return
            return self._send_bytes(render_plot_editor_page(root, root / rel))

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

    def do_DELETE(self):
        parsed = urlparse(self.path)
        name = parsed.path.removeprefix("/api/plots/")
        if not parsed.path.startswith("/api/plots/") or not re.fullmatch(r"[0-9a-f]{32}\.png", name):
            self.send_error(404)
            return
        try:
            image = self._root_dir() / ".plot-cache" / name
            image.unlink()
            image.with_suffix(".json").unlink(missing_ok=True)
        except FileNotFoundError:
            self.send_error(404)
            return
        self._send_json({"deleted": True})

    def do_POST(self):
        endpoint = urlparse(self.path).path
        if endpoint not in ("/api/plots", "/api/plots/search"):
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 8192:
                raise ValueError("Invalid request size")
            request = json.loads(self.rfile.read(length))
            if endpoint == "/api/plots/search":
                if not isinstance(request, dict):
                    raise ValueError("Enter a valid email address")
                email = normalize_email(request.get("email"))
                return self._send_json({"plots": plots_for_email(self._root_dir() / ".plot-cache", email)})
            if not isinstance(request, dict) or not isinstance(request.get("event"), str):
                raise ValueError("Select an event")
            events = scan_events(self._root_dir())
            if request["event"] not in {event["path"] for event in events}:
                raise ValueError("Unknown event")
            event_path = (self._root_dir() / request["event"]).resolve()
            normalized = render_plot(event_path, request)
            cache = self._root_dir() / ".plot-cache"
            cleanup_cache(cache)
            name = store_plot(cache, normalized, normalize_email(request.get("email")), request["event"], request.get("title", ""))
            url = f"/generated/{name}"
            result = {"url": url, "expires_in_seconds": TTL_SECONDS, "delete_url": f"/api/plots/{name}"}
            return self._send_json(result)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            payload = json.dumps({"error": str(exc)}).encode("utf-8")
            self.send_response(400)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

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
            entries = [Path(path) / name for name in os.listdir(path) if name != ".plot-cache"]
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
