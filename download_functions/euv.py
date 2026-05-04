import datetime
import csv
import io
import os
import re
from pathlib import Path

import pandas as pd
import requests

from DataManager import DataManager


LASP_SOHO_SEM_BASE_URL = (
    "https://lasp.colorado.edu/eve/data_access/eve_data/"
    "lasp_soho_sem_data/long/15_sec_avg"
)
CDAWEB_HAPI_DATA_URL = "https://cdaweb.gsfc.nasa.gov/hapi/data"
CDAWEB_SOHO_SEM_DATASET = "SOHO_CELIAS-SEM_15S"
CDAWEB_SOHO_SEM_PARAMETERS = "first_order_flux,central_order_flux"
SOHO_SEM_FILL_THRESHOLD = -1.0e30
_HAPI_UNIT_SUFFIX_RE = re.compile(r"^([+-]?\d+(?:\.\d+)?[Ee][+-]?\d{1,3})2e$")


def _parse_hapi_float(value: object) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    match = _HAPI_UNIT_SUFFIX_RE.match(text)
    if match:
        text = match.group(1)
    return float(text)


def _validate_soho_sem_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        raise Exception("No SOHO SEM data rows")

    required_columns = {"flux_26_34", "flux_01_50"}
    missing = required_columns - set(df.columns)
    if missing:
        raise Exception(f"Missing SOHO SEM columns: {', '.join(sorted(missing))}")

    df = df.copy()
    df.index = pd.to_datetime(df.index)
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        df.loc[df[column] <= SOHO_SEM_FILL_THRESHOLD, column] = pd.NA

    df = df.dropna(subset=sorted(required_columns), how="any")
    if df.empty:
        raise Exception("No valid SOHO SEM flux values")
    return df


def _write_soho_sem_csv(df: pd.DataFrame, temp_path: Path) -> Path:
    df = _validate_soho_sem_frame(df)
    temp_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(temp_path, index=True)
    if not temp_path.exists() or temp_path.stat().st_size == 0:
        temp_path.unlink(missing_ok=True)
        raise Exception(f"Temporary file is empty: {temp_path}")
    return temp_path


def _download_soho_sem_lasp(
    date: datetime.date,
    timeout: int | float = 60,
) -> pd.DataFrame:
    year = date.strftime("%Y")
    filename_remote = f"{date.strftime('%y_%m_%d')}_v4.00"
    url = f"{LASP_SOHO_SEM_BASE_URL}/{year}/{filename_remote}"

    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}")

    rows = []
    base_time = datetime.datetime.combine(date, datetime.datetime.min.time())
    for line in response.text.splitlines():
        line = line.strip()
        if not line or line.startswith(";"):
            continue

        parts = line.split()
        if len(parts) < 14:
            continue

        try:
            seconds = float(parts[3])
            flux_26_34 = float(parts[12])
            flux_01_50 = float(parts[13])
        except ValueError:
            continue

        if flux_26_34 == -1.0 or flux_01_50 == -1.0:
            continue

        rows.append(
            {
                "time": base_time + datetime.timedelta(seconds=seconds),
                "flux_26_34": flux_26_34,
                "flux_01_50": flux_01_50,
            }
        )

    if not rows:
        raise Exception("empty LASP response")

    df = pd.DataFrame(rows)
    df.set_index("time", inplace=True)
    return df


def _download_soho_sem_cdaweb(
    date: datetime.date,
    timeout: int | float = 120,
) -> pd.DataFrame:
    start = datetime.datetime.combine(date, datetime.time.min)
    stop = start + datetime.timedelta(days=1)
    verify_ssl = os.environ.get("SOHO_SEM_CDAWEB_VERIFY_SSL", "true").lower() not in {
        "0",
        "false",
        "no",
    }
    response = requests.get(
        CDAWEB_HAPI_DATA_URL,
        params={
            "id": CDAWEB_SOHO_SEM_DATASET,
            "parameters": CDAWEB_SOHO_SEM_PARAMETERS,
            "time.min": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "time.max": stop.strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        headers={"Accept": "text/csv"},
        timeout=timeout,
        verify=verify_ssl,
    )
    if response.status_code != 200:
        raise Exception(f"HTTP {response.status_code}: {response.text[:200]}")

    rows = []
    if response.text.lstrip().startswith("{"):
        payload = response.json()
        status = payload.get("status", {})
        status_code = status.get("code")
        if status_code not in (None, 1200):
            raise Exception(f"status {status_code}: {status.get('message', 'unknown error')}")

        source_rows = payload.get("data") or []
    else:
        source_rows = csv.reader(io.StringIO(response.text))

    for row in source_rows:
        if len(row) < 3:
            continue
        rows.append({
            "time": pd.to_datetime(row[0]),
            "flux_26_34": _parse_hapi_float(row[1]),
            "flux_01_50": _parse_hapi_float(row[2]),
        })

    if not rows:
        raise Exception("empty CDAWeb response")

    df = pd.DataFrame(rows)
    df.set_index("time", inplace=True)
    return df


def download_soho_sem(
    date: datetime.date,
    data_manager: "DataManager",
    **kwargs,
) -> Path:
    filename = kwargs.get("filename", f"soho_sem_{date.strftime('%Y%m%d')}.csv")
    final_path = data_manager.get_download_path("soho_sem", date, filename, create_dir=False)

    temp_path = kwargs.get("temp_path")
    if not temp_path:
        temp_path = final_path.with_suffix(".tmp")
    temp_path = Path(temp_path)

    if not kwargs.get("force_redownload", False) and final_path.exists():
        try:
            df_test = pd.read_csv(final_path, nrows=1)
            if not df_test.empty:
                return final_path
            print(f"   SOHO SEM file is empty, redownloading: {final_path}")
        except Exception as error:
            print(f"   SOHO SEM file is invalid ({error}), redownloading: {final_path}")

    print(f"Загрузка SOHO SEM данных за {date}...")
    source_errors = []
    for source_name, loader in (
        ("LASP ASCII", _download_soho_sem_lasp),
        ("NASA CDAWeb HAPI", _download_soho_sem_cdaweb),
    ):
        try:
            df = loader(date)
            print(f"   SOHO SEM источник: {source_name}")
            return _write_soho_sem_csv(df, temp_path)
        except requests.exceptions.Timeout:
            error = f"{source_name}: timeout"
        except requests.exceptions.ConnectionError:
            error = f"{source_name}: connection error"
        except Exception as source_error:
            error = f"{source_name}: {source_error}"

        source_errors.append(error)
        print(f"   SOHO SEM {error}")

    if temp_path.exists():
        temp_path.unlink(missing_ok=True)
    raise Exception(f"Ошибка загрузки SOHO SEM данных за {date}: {'; '.join(source_errors)}")
