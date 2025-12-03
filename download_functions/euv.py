import pandas as pd
import datetime
import requests
import os
from pathlib import Path
from DataManager import DataManager

def download_soho_sem(
    date: datetime.date,
    data_manager: 'DataManager',
    **kwargs
) -> Path:  # Возвращаем путь к файлу (Path)
    
    # Получаем путь для финального файла
    filename = kwargs.get('filename', f"soho_sem_{date.strftime('%Y%m%d')}.csv")
    final_path = data_manager.get_download_path('soho_sem', date, filename)
    
    # Получаем путь для временного файла (из kwargs или создаем)
    temp_path = kwargs.get('temp_path', None)
    if not temp_path:
        temp_path = final_path.with_suffix('.tmp')
    
    # Проверяем, существует ли уже готовый файл
    if not kwargs.get('force_redownload', False) and final_path.exists():
        try:
            # Проверяем, что файл можно прочитать (валидный)
            df_test = pd.read_csv(final_path, nrows=1)
            if not df_test.empty:
                return final_path
            else:
                print(f"⚠️ Файл {final_path} пустой, перезагружаем...")
        except Exception as e:
            print(f"⚠️ Файл {final_path} поврежден ({e}), перезагружаем...")
    
    try:
        print(f"📡 Загрузка SOHO SEM данных за {date}...")
        
        year = date.strftime('%Y')
        yy = date.strftime('%y')
        mm = date.strftime('%m')
        dd = date.strftime('%d')
        
        filename_remote = f'{yy}_{mm}_{dd}_v4.00'
        url = f'https://lasp.colorado.edu/eve/data_access/eve_data/lasp_soho_sem_data/long/15_sec_avg/{year}/{filename_remote}'
        
        print(f"   📥 URL: {url}")
        
        response = requests.get(url, timeout=60)
        
        if response.status_code != 200:
            raise Exception(f"Не удалось скачать файл: HTTP {response.status_code}")
        
        print(f"   ✅ Файл получен ({len(response.text)} байт)")
        
        data = []
        base_time = datetime.datetime.combine(date, datetime.datetime.min.time())

        lines_processed = 0
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
                    lines_processed += 1
                except ValueError as e:
                    continue
        
        if not data:
            raise Exception("Нет данных в файле")
        
        print(f"   📊 Обработано строк: {lines_processed}, записей: {len(data)}")
        
        df = pd.DataFrame(data)
        df.set_index('time', inplace=True)

        print(f"   💾 Сохранение во временный файл: {temp_path}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(temp_path, index=True)

        if not temp_path.exists():
            raise Exception(f"Не удалось создать временный файл: {temp_path}")
        
        file_size = temp_path.stat().st_size
        if file_size == 0:
            temp_path.unlink()
            raise Exception(f"Временный файл пустой: {temp_path}")
        
        print(f"   ✅ Данные сохранены ({file_size} байт)")

        return temp_path
        
    except requests.exceptions.Timeout:
        raise Exception(f"Таймаут при загрузке SOHO SEM данных за {date}")
    except requests.exceptions.ConnectionError:
        raise Exception(f"Ошибка соединения при загрузке SOHO SEM данных за {date}")
    except Exception as e:
        # Если есть временный файл - удаляем его
        if 'temp_path' in locals() and temp_path.exists():
            try:
                temp_path.unlink()
                print(f"🧹 Удален временный файл после ошибки: {temp_path}")
            except:
                pass
        
        error_msg = f"Ошибка загрузки SOHO SEM данных за {date}: {str(e)}"
        print(f"❌ {error_msg}")
        raise Exception(error_msg)