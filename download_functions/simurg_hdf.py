import datetime
import time
import requests
from DataManager import DataManager
from pathlib import Path

SIMURG_CHUNK_SIZE = 1024 * 1024
SIMURG_PROGRESS_INTERVAL_SECONDS = 10
SIMURG_MAX_ATTEMPTS = 5


def _format_bytes(size: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} TB"


def _timeout_from_kwargs(kwargs):
    if "timeout" in kwargs:
        return kwargs["timeout"]
    return None


def _content_length(response) -> int:
    try:
        return int(response.headers.get("content-length") or 0)
    except (TypeError, ValueError):
        return 0


def _supports_resume(response) -> bool:
    return response.status_code == 206 or "bytes" in response.headers.get("content-range", "").lower()


def download_simurg_hdf(
    date: datetime.date,
    data_manager: 'DataManager',
    **kwargs
) -> Path:
    """Скачать данные Simurg в формате HDF5 за указанную дату"""
    
    data_type = kwargs.get('data_type', 'obs')  # 'obs', 'f107', 'fism2', 'mgnm'
    timeout = _timeout_from_kwargs(kwargs)
    progress_interval = kwargs.get("progress_interval", SIMURG_PROGRESS_INTERVAL_SECONDS)
    chunk_size = kwargs.get("chunk_size", SIMURG_CHUNK_SIZE)
    max_attempts = kwargs.get("max_attempts", SIMURG_MAX_ATTEMPTS)

    filename = f"simurg_{data_type}_{date.strftime('%Y%m%d')}.h5"
    final_path = data_manager.get_download_path('simurg_hdf', date, filename)

    temp_path = kwargs.get('temp_path', None)
    if not temp_path:
        temp_path = final_path.with_suffix('.h5.tmp')
    
    if not kwargs.get('force_redownload', False) and final_path.exists():
        try:
            import h5py
            with h5py.File(final_path, 'r') as f:
                if len(f.keys()) > 0:
                    return final_path
                else:
                    print(f"⚠️ Файл {final_path} пустой или поврежден, перезагружаем...")
        except Exception as e:
            print(f"⚠️ Файл {final_path} поврежден ({e}), перезагружаем...")
    
    try:
        date_str = date.strftime('%Y-%m-%d')
        url = f"https://simurg.iszf.irk.ru/gen_file?data={data_type}&date={date_str}"

        print(f"📡 Загрузка Simurg HDF ({data_type}) за {date}...")
        
        # Создаем директорию для временного файла
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        total_size = 0
        for attempt in range(1, max_attempts + 1):
            resume_from = temp_path.stat().st_size if temp_path.exists() else 0
            headers = {"Range": f"bytes={resume_from}-"} if resume_from > 0 else {}
            mode = "ab" if resume_from > 0 else "wb"
            if resume_from > 0:
                print(f"   Продолжаю с {_format_bytes(resume_from)} (попытка {attempt}/{max_attempts})")

            try:
                with requests.get(url, timeout=timeout, stream=True, headers=headers) as response:
                    response.raise_for_status()

                    if resume_from > 0 and not _supports_resume(response):
                        print("   Сервер не поддержал докачку, начинаю загрузку заново.")
                        resume_from = 0
                        mode = "wb"

                    response_size = _content_length(response)
                    if response_size > 0:
                        total_size = resume_from + response_size
                        print(f"   Размер: {_format_bytes(total_size)}")

                    downloaded = resume_from
                    last_progress_at = time.monotonic()
                    with open(temp_path, mode) as f:
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            now = time.monotonic()
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if now - last_progress_at >= progress_interval:
                                    if total_size > 0:
                                        percent = downloaded / total_size * 100
                                        print(
                                            f"   Загружено {_format_bytes(downloaded)} / "
                                            f"{_format_bytes(total_size)} ({percent:.1f}%)"
                                        )
                                    else:
                                        print(f"   Загружено {_format_bytes(downloaded)}")
                                    last_progress_at = now

                break
            except requests.exceptions.RequestException as e:
                if attempt >= max_attempts:
                    raise
                current_size = temp_path.stat().st_size if temp_path.exists() else 0
                print(
                    f"   Соединение оборвалось: {e}. "
                    f"Сохранено {_format_bytes(current_size)}, повторяю..."
                )
                continue
        
        # Проверяем, что временный файл создан и не пустой
        if not temp_path.exists():
            raise Exception(f"Не удалось создать временный файл: {temp_path}")
        
        file_size = temp_path.stat().st_size
        if file_size == 0:
            temp_path.unlink()  # Удаляем пустой файл
            raise Exception(f"Временный файл пустой: {temp_path}")
        print(f"   Simurg HDF загружен: {_format_bytes(file_size)}")
        return temp_path
        
    except requests.exceptions.Timeout as e:
        partial_hint = ""
        if 'temp_path' in locals() and temp_path.exists():
            partial_hint = f" Частичный файл сохранён для докачки: {temp_path}"
        raise Exception(f"Таймаут при загрузке Simurg HDF за {date}: {e}.{partial_hint}")
        
    except requests.exceptions.RequestException as e:
        partial_hint = ""
        if 'temp_path' in locals() and temp_path.exists():
            partial_hint = f" Частичный файл сохранён для докачки: {temp_path}"
        raise Exception(f"Ошибка сети при загрузке Simurg HDF за {date}: {e}.{partial_hint}")
        
    except Exception as e:
        error_msg = f"Ошибка загрузки Simurg HDF ({data_type}) за {date}: {str(e)}"
        raise Exception(error_msg)
