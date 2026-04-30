import datetime
import time
from urllib.error import ContentTooShortError, HTTPError, URLError
from urllib.request import urlretrieve
from DataManager import DataManager
from pathlib import Path

SIMURG_PROGRESS_INTERVAL_SECONDS = 10


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


def download_simurg_hdf(
    date: datetime.date,
    data_manager: 'DataManager',
    **kwargs
) -> Path:
    """Скачать данные Simurg в формате HDF5 за указанную дату"""
    
    data_type = kwargs.get('data_type', 'obs')  # 'obs', 'f107', 'fism2', 'mgnm'
    progress_interval = kwargs.get("progress_interval", SIMURG_PROGRESS_INTERVAL_SECONDS)

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
        if temp_path.exists():
            temp_path.unlink()

        last_progress_at = 0.0

        def _reporthook(block_count: int, block_size: int, total_size: int):
            nonlocal last_progress_at
            now = time.monotonic()
            if block_count == 0 and total_size > 0:
                print(f"   Размер: {_format_bytes(total_size)}")
                return
            if now - last_progress_at < progress_interval:
                return
            downloaded = block_count * block_size
            if total_size > 0:
                downloaded = min(downloaded, total_size)
                percent = downloaded / total_size * 100
                print(
                    f"   Загружено {_format_bytes(downloaded)} / "
                    f"{_format_bytes(total_size)} ({percent:.1f}%)"
                )
            else:
                print(f"   Загружено {_format_bytes(downloaded)}")
            last_progress_at = now

        urlretrieve(url, temp_path, reporthook=_reporthook)
        
        # Проверяем, что временный файл создан и не пустой
        if not temp_path.exists():
            raise Exception(f"Не удалось создать временный файл: {temp_path}")
        
        file_size = temp_path.stat().st_size
        if file_size == 0:
            temp_path.unlink()  # Удаляем пустой файл
            raise Exception(f"Временный файл пустой: {temp_path}")
        print(f"   Simurg HDF загружен: {_format_bytes(file_size)}")
        return temp_path
        
    except (ContentTooShortError, HTTPError, URLError) as e:
        raise Exception(f"Ошибка сети при загрузке Simurg HDF за {date}: {e}")
        
    except Exception as e:
        error_msg = f"Ошибка загрузки Simurg HDF ({data_type}) за {date}: {str(e)}"
        raise Exception(error_msg)
