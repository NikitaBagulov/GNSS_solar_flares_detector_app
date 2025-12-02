import pandas as pd
import datetime
import requests
import os
from DataManager import DataManager

def download_soho_sem(
    date: datetime.date,
    data_manager: 'DataManager',
    **kwargs
) -> pd.DataFrame:

    filename = kwargs.get('filename', f"soho_sem_{date.strftime('%Y%m%d')}.csv")

    if not kwargs.get('force_redownload', False) and \
       data_manager.check_file_exists('soho_sem', date, filename):
        
        file_path = data_manager.get_download_path('soho_sem', date, filename)
        return file_path
    
    try:
        year = date.strftime('%Y')
        yy = date.strftime('%y')
        mm = date.strftime('%m')
        dd = date.strftime('%d')
        
        filename_remote = f'{yy}_{mm}_{dd}_v4.00'
        url = f'https://lasp.colorado.edu/eve/data_access/eve_data/lasp_soho_sem_data/long/15_sec_avg/{year}/{filename_remote}'

        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            raise Exception(f"Не удалось скачать файл: HTTP {response.status_code}")

        data = []
        base_time = datetime.datetime.combine(date, datetime.datetime.min.time())

        for line in response.text.split('\n'):
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            parts = line.split()
            if len(parts) >= 14:
                try:
                    seconds = float(parts[3])
                    flux1 = float(parts[12])  # flux_26_34
                    flux2 = float(parts[13])  # flux_01_50

                    if flux1 == -1.0 or flux2 == -1.0:
                        continue
                    
                    time_val = base_time + datetime.timedelta(seconds=seconds)
                    data.append({
                        'time': time_val,
                        'flux_26_34': flux1,
                        'flux_01_50': flux2
                    })
                except:
                    continue

        if not data:
            raise Exception("Нет данных в файле")
        
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)

        file_path = data_manager.get_download_path('soho_sem', date, filename)
        df.to_csv(file_path, index=True)     
        return file_path
        
    except Exception as e:
        raise