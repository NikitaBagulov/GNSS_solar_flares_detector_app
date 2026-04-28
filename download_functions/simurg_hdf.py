import datetime
import requests
from DataManager import DataManager
from pathlib import Path

SIMURG_CHUNK_SIZE = 64 * 1024 * 1024


def download_simurg_hdf(
    date: datetime.date,
    data_manager: 'DataManager',
    **kwargs
) -> Path:
    """Скачать данные Simurg в формате HDF5 за указанную дату"""
    
    data_type = kwargs.get('data_type', 'obs')  # 'obs', 'f107', 'fism2', 'mgnm'
    timeout = kwargs.get('timeout', 120)

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
                    file_size = final_path.stat().st_size
                    print(f"   ⏭️ Файл Simurg HDF уже существует: {final_path} ({file_size / 1024:.1f} KB)")
                    return final_path
                else:
                    print(f"⚠️ Файл {final_path} пустой или поврежден, перезагружаем...")
        except Exception as e:
            print(f"⚠️ Файл {final_path} поврежден ({e}), перезагружаем...")
    
    try:
        date_str = date.strftime('%Y-%m-%d')
        url = f"https://simurg.iszf.irk.ru/gen_file?data={data_type}&date={date_str}"

        print(f"📡 Загрузка Simurg HDF ({data_type}) за {date}...")
        print(f"   📥 URL: {url}")
        
        # Создаем директорию для временного файла
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        
        with requests.get(url, timeout=timeout, stream=True) as response:
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            print(f"   📊 Размер файла: {total_size / 1024:.1f} KB" if total_size > 0 else "   📊 Размер неизвестен")

            downloaded = 0
            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=SIMURG_CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
        
        print(f"   ✅ Загрузка завершена: {downloaded / 1024:.1f} KB")
        
        # Проверяем, что временный файл создан и не пустой
        if not temp_path.exists():
            raise Exception(f"Не удалось создать временный файл: {temp_path}")
        
        file_size = temp_path.stat().st_size
        if file_size == 0:
            temp_path.unlink()  # Удаляем пустой файл
            raise Exception(f"Временный файл пустой: {temp_path}")
        return temp_path
        
    except requests.exceptions.Timeout as e:
        # Удаляем временный файл при таймауте
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        raise Exception(f"Таймаут при загрузке Simurg HDF за {date}: {e}")
        
    except requests.exceptions.RequestException as e:
        # Удаляем временный файл при ошибке сети
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        raise Exception(f"Ошибка сети при загрузке Simurg HDF за {date}: {e}")
        
    except Exception as e:
        # Удаляем временный файл при любой другой ошибке
        if 'temp_path' in locals() and temp_path.exists():
            temp_path.unlink()
        error_msg = f"Ошибка загрузки Simurg HDF ({data_type}) за {date}: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)
