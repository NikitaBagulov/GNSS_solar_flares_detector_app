import pandas as pd
import datetime
import requests
import os
from DataManager import DataManager
from pathlib import Path

import pandas as pd
import datetime
import requests
import os
from DataManager import DataManager
from pathlib import Path

def download_simurg_hdf(
    date: datetime.date,
    data_manager: 'DataManager',
    **kwargs
) -> Path:
    data_type = kwargs.get('data_type', 'obs')  # 'obs', 'f107', 'fism2', 'mgnm'
    timeout = kwargs.get('timeout', 60)

    filename = f"simurg_{data_type}_{date.strftime('%Y%m%d')}.h5"

    file_path = data_manager.get_download_path('simurg', date, filename)
    
    if file_path.exists() and not kwargs.get('force_redownload', False):
        #f"Файл уже существует: {file_path}"
        return file_path
    
    try:
        date_str = date.strftime('%Y-%m-%d')
        url = f"https://simurg.iszf.irk.ru/gen_file?data={data_type}&date={date_str}"

        print(f"Скачивание Simurg HDF: {url}")
        
        response = requests.get(url, timeout=timeout, stream=True)
        response.raise_for_status()

        with open(file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=65536):
                f.write(chunk)
        
        print(f"Файл сохранен: {file_path} ({file_path.stat().st_size / 1024:.1f} KB)")
        
        return file_path
        
    except requests.exceptions.Timeout as e:
        raise e
    except requests.exceptions.RequestException as e:
        raise e
    except Exception as e:
        raise e