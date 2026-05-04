from sunpy.net import Fido, attrs as a
from sunpy.time import TimeRange
from datetime import date, datetime, timedelta
from typing import Optional, List, Dict, Any
import pandas as pd
from pathlib import Path
from DataManager import DataManager

GOES_SATELLITES = {
    13: {"start": "2006-04-01", "end": "2020-12-31"},
    14: {"start": "2009-07-01", "end": "2023-01-01"},
    15: {"start": "2010-03-01", "end": "2022-03-01"},
    16: {"start": "2017-01-01", "end": "2023-10-01"},
    17: {"start": "2018-02-01", "end": "2023-10-01"},
    18: {"start": "2022-03-01", "end": "2030-01-01"},
}

def get_available_satellites(
    start_time: datetime,
    end_time: datetime,
    prefer_order: List[int] = None
) -> List[int]:
    if prefer_order is None:
        prefer_order = [18, 17, 16, 15, 14, 13]
    
    available = []
    
    for sat_num in prefer_order:
        if sat_num in GOES_SATELLITES:
            sat_start = datetime.strptime(GOES_SATELLITES[sat_num]["start"], "%Y-%m-%d")
            sat_end = datetime.strptime(GOES_SATELLITES[sat_num]["end"], "%Y-%m-%d")
            
            if start_time <= sat_end and end_time >= sat_start:
                available.append(sat_num)
    
    return available

def download_goes_xray(
    date: date,
    data_manager: 'DataManager',
    **kwargs
) -> pd.DataFrame:
    filename = f"goes_{date.strftime('%Y%m%d')}.csv"
    final_path = data_manager.get_download_path('goes_xray', date, filename, create_dir=False)
    
    temp_path = kwargs.get('temp_path', None)
    if not temp_path:
        temp_path = final_path.with_suffix('.tmp')
    
    if not kwargs.get('force_redownload', False) and final_path.exists():
        try:
            return final_path
        except:
            pass
    
    try:
        start = datetime.combine(date, datetime.min.time())
        end = datetime.combine(date, datetime.max.time())

        sats = get_available_satellites(start, end)
        
        for sat in sats:
            try:
                results = Fido.search(
                    a.Time(start, end),
                    a.Instrument.xrs,
                    a.goes.SatelliteNumber(sat)
                )
                
                if len(results) > 0:
                    files = Fido.fetch(results[0])
                    
                    if len(files) > 0:
                        from sunpy.timeseries import TimeSeries
                        ts = TimeSeries(files[0])
                        df = ts.to_dataframe()

                        temp_path.parent.mkdir(parents=True, exist_ok=True)
                        df.to_csv(temp_path)

                        return temp_path
                        
            except Exception as e:
                print(f"GOES-{sat} не удался: {e}")
                continue

        return None
        
    except Exception as e:
        print(f"❌ Ошибка загрузки GOES X-ray: {e}")
        return None
