import numpy as np
import math

# Радиус Земли в метрах
RE_meters = 6371000

def great_circle_distance_vec(lat, lon, lat0=0.0, lon0=0.0):
    lat = np.radians(lat)
    lon = np.radians(lon)
    lat0 = np.radians(lat0)
    lon0 = np.radians(lon0)

    dlon = lon - lon0
    dlon = (dlon + np.pi) % (2 * np.pi) - np.pi

    cosgamma = (
        np.sin(lat) * np.sin(lat0)
        + np.cos(lat) * np.cos(lat0) * np.cos(dlon)
    )

    cosgamma = np.clip(cosgamma, -1.0, 1.0)
    return RE_meters * np.arccos(cosgamma)


def great_circle_distance_rad(late, lone, latp, lonp, R=RE_meters):
    late, lone, latp, lonp = map(math.radians, [late, lone, latp, lonp])
    dlon = lonp - lone
    dlon = (dlon + math.pi) % (2 * math.pi) - math.pi  # Нормализация

    cosgamma = math.sin(late) * math.sin(latp) + math.cos(late) * math.cos(latp) * math.cos(dlon)
    cosgamma = min(1, max(-1, cosgamma))  # Защита от ошибок округления
    return R * math.acos(cosgamma)


def is_daytime(lat_point, lon_point, observer_lat=0.0, observer_lon=0.0):
    dist = great_circle_distance_rad(lat_point, lon_point, observer_lat, observer_lon)
    # Условно: если расстояние до "дневного центра" < половина радиуса Земли
    return dist < (math.pi / 2 * RE_meters)


def compute_day_night_index(dates, time_key):
    if not dates:
        return 0.0

    # распаковка
    data = np.array(dates, dtype=float)
    lat = data[:, 0]
    lon = data[:, 1]
    values = data[:, 2]

    # расстояния
    distances = great_circle_distance_vec(lat, lon)

    # день / ночь
    day_mask = distances < (np.pi / 2 * RE_meters)
    night_mask = ~day_mask

    # --- день ---
    if np.any(day_mask):
        d_day = distances[day_mask]
        v_day = values[day_mask]
        weights = 1 - d_day / ((2 * np.pi * RE_meters) / 4)
        I_day = np.nan_to_num(weights * v_day)
        total_day = np.sum(I_day)
    else:
        total_day = 0.0

    # --- ночь ---
    if np.any(night_mask):
        v_night = values[night_mask]
        total_night = np.sum(np.nan_to_num(v_night))
    else:
        total_night = 0.0

    # итог
    if total_night == 0:
        return float(total_day)

    return float(total_day / total_night)


