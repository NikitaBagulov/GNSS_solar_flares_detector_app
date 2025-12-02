import pandas as pd
from datetime import datetime, date
import requests
import os
from DataManager import DataManager
from pathlib import Path
from sunpy.net import Fido, attrs as a
from datetime import date, datetime
import pandas as pd
from astropy.time import Time

def download_flares(
    date: date,
    data_manager: 'DataManager',
    **kwargs
) -> pd.DataFrame:
    filename = f"flares_{date.strftime('%Y%m%d')}.csv"
    file_path = data_manager.get_download_path('flares', date, filename)

    if not kwargs.get('force_redownload', False) and file_path.exists():
        try:
            return file_path
        except:
            pass
    
    try:
        start = datetime.combine(date, datetime.min.time())
        end = datetime.combine(date, datetime.max.time())

        tstart = start.strftime('%Y/%m/%d 00:00')
        tend = end.strftime('%Y/%m/%d 23:59')

        result = Fido.search(
            a.Time(tstart, tend),
            a.hek.EventType("FL")
        )
        
        if len(result) == 0:
            return pd.DataFrame()

        hek_results = result['hek']
        
        flares_data = []
        for flare in hek_results:
            try:
                flare_class = flare.get('fl_goescls', '').strip()
                
                if not flare_class:
                    continue
                
                flares_data.append({
                    'class': flare_class,
                    'start_time': Time(flare['event_starttime']).to_datetime(),
                    'peak_time': Time(flare['event_peaktime']).to_datetime(),
                    'end_time': Time(flare['event_endtime']).to_datetime(),
                    'hpc_x': flare.get('hpc_x'),
                    'hpc_y': flare.get('hpc_y')
                })
            except:
                continue

        df = pd.DataFrame(flares_data)

        if not df.empty:
            df.to_csv(file_path, index=False)
        
        return file_path
        
    except Exception as e:
        return f"Ошибка скачивания вспышек: {e}"