from DataManager import DataManager
from datetime import date
from download_functions.euv import download_soho_sem
from download_functions.simurg_hdf import download_simurg_hdf
from download_functions.xray import download_goes_xray
from download_functions.hek_flares import download_flares

# from pprint import pprint

data_manager = DataManager()
data_manager.register_download_function("soho_sem", download_func=download_soho_sem)
data_manager.register_download_function("goes_xray", download_func=download_goes_xray)
data_manager.register_download_function("hek_flares", download_func=download_flares)
# data_manager.register_download_function("simurg_hdf", download_func=download_simurg_hdf)

# res = data_manager.download_by_date(target_date=date(2024, 5, 10))
# pprint(res)

from FlareTracker import FlareTracker

tracker = FlareTracker(
    data_manager=data_manager,
    min_year=2024,
    min_flare_class="X1.0"
)

# 2. Проверяем пропущенные дни и скачиваем
print("Проверка пропущенных данных...")
tracker.download_missed_data()