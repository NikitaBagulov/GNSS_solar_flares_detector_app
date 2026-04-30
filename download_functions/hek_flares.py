import pandas as pd
from datetime import datetime, date, timedelta
import requests
import os
from DataManager import DataManager
from pathlib import Path
from sunpy.net import Fido, attrs as a
from astropy.time import Time

def download_flares(
    date: date,
    data_manager: 'DataManager',
    **kwargs
) -> Path:  # Теперь возвращаем Path к файлу
    """Скачать данные о вспышках за указанную дату из HEK"""
    
    # Получаем путь для финального файла
    filename = f"flares_{date.strftime('%Y%m%d')}.csv"
    final_path = data_manager.get_download_path('flares', date, filename)
    
    # Получаем путь для временного файла
    temp_path = kwargs.get('temp_path', None)
    if not temp_path:
        temp_path = final_path.with_suffix('.tmp')
    
    # Проверяем, существует ли уже готовый файл
    if not kwargs.get('force_redownload', False) and final_path.exists():
        try:
            # Проверяем, что файл можно прочитать и он не пустой
            df_test = pd.read_csv(final_path, nrows=1)
            if not df_test.empty:
                return final_path
            else:
                print(f"⚠️ Файл {final_path} пустой, перезагружаем...")
        except Exception as e:
            print(f"⚠️ Файл {final_path} поврежден ({e}), перезагружаем...")
    
    try:
        print(f"📡 Загрузка данных о вспышках за {date}...")
        
        start = datetime.combine(date, datetime.min.time())
        end = datetime.combine(date, datetime.max.time())

        tstart = start.strftime('%Y/%m/%d 00:00')
        tend = end.strftime('%Y/%m/%d 23:59')

        result = Fido.search(
            a.Time(tstart, tend),
            a.hek.EventType("FL")
        )
        
        if len(result) == 0:
            print(f"   📭 Вспышек за {date} не найдено")
            # Создаем пустой DataFrame и сохраняем его
            empty_df = pd.DataFrame(columns=[
                'class', 'start_time', 'peak_time', 'end_time', 
                'hpc_x', 'hpc_y', 'flare_value', 'date'
            ])
            
            # Сохраняем во временный файл
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            empty_df.to_csv(temp_path, index=False)
            
            return temp_path

        hek_results = result['hek']
        print(f"   ✅ Найдено {len(hek_results)} вспышек")
        
        flares_data = []
        for flare in hek_results:
            try:
                flare_class = flare.get('fl_goescls', '').strip()
                
                if not flare_class:
                    continue
                
                start_time = Time(flare['event_starttime']).to_datetime()
                peak_time = Time(flare['event_peaktime']).to_datetime()
                end_time = Time(flare['event_endtime']).to_datetime()
                
                # Конвертируем класс вспышки в числовое значение
                flare_value = _flare_class_to_value(flare_class)
                
                flares_data.append({
                    'class': flare_class,
                    'start_time': start_time,
                    'peak_time': peak_time,
                    'end_time': end_time,
                    'hpc_x': flare.get('hpc_x'),
                    'hpc_y': flare.get('hpc_y'),
                    'flare_value': flare_value,
                    'date': date.strftime('%Y-%m-%d')  # Добавляем дату для удобства
                })
            except Exception as e:
                print(f"      ⚠️ Ошибка обработки вспышки: {e}")
                continue

        if not flares_data:
            print(f"   📭 Нет валидных данных о вспышках")
            # Создаем пустой DataFrame
            empty_df = pd.DataFrame(columns=[
                'class', 'start_time', 'peak_time', 'end_time', 
                'hpc_x', 'hpc_y', 'flare_value', 'date'
            ])
            
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            empty_df.to_csv(temp_path, index=False)
            return temp_path
        
        df = pd.DataFrame(flares_data)
        
        # Сортируем по времени начала
        df = df.sort_values('start_time')
        
        # Сохраняем во ВРЕМЕННЫЙ файл
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(temp_path, index=False)
        
        # Проверяем, что временный файл создан
        if not temp_path.exists():
            raise Exception(f"Не удалось создать временный файл: {temp_path}")
        
        # Возвращаем путь к временному файлу
        return temp_path
        
    except Exception as e:
        # Если есть временный файл - удаляем его
        if 'temp_path' in locals() and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass
        
        error_msg = f"Ошибка загрузки данных о вспышках за {date}: {str(e)}"
        raise Exception(error_msg)

def _flare_class_to_value(flare_class: str) -> float:
    """Конвертирует класс вспышки в числовое значение"""
    if not isinstance(flare_class, str):
        return 0.0
    
    flare_class = flare_class.strip().upper()
    
    # Множители для классов
    multipliers = {
        'X': 10.0,
        'M': 1.0,
        'C': 0.1,
        'B': 0.01,
        'A': 0.001
    }
    
    if not flare_class:
        return 0.0
    
    letter = flare_class[0]
    number_part = flare_class[1:] if len(flare_class) > 1 else '1.0'
    
    try:
        number = float(number_part)
    except ValueError:
        number = 1.0
    
    return multipliers.get(letter, 0.0) * number
