import numpy as np
import math
import datetime

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

    # ВАЖНО: clip не лечит nan, поэтому nan надо убрать до arccos уровнем выше.
    cosgamma = np.clip(cosgamma, -1.0, 1.0)
    return RE_meters * np.arccos(cosgamma)


def _to_utc_datetime(dt: datetime.datetime) -> datetime.datetime:
    """Приводим datetime к UTC. Если naive — считаем, что это уже UTC."""
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def _subsolar_point(dt: datetime.datetime):
    """
    Приближение подсолнечной точки:
    - широта = солнечная деклинация (declination)
    - долгота = 180° - GHA (по GMST и RA)
    Точности достаточно для разделения день/ночь и весов по расстоянию.
    """
    dt = _to_utc_datetime(dt)

    # Julian Day
    # алгоритм: https://aa.usno.navy.mil/faq/JD_formula (классический)
    year, month = dt.year, dt.month
    day = dt.day + (dt.hour + (dt.minute + dt.second / 60.0) / 60.0) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + (A // 4)
    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5

    T = (JD - 2451545.0) / 36525.0  # centuries since J2000.0

    # Sun mean longitude (deg), mean anomaly (deg)
    L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T * T) % 360.0
    M = (357.52911 + 35999.05029 * T - 0.0001537 * T * T) % 360.0

    # Equation of center (deg)
    Mrad = math.radians(M)
    C = (1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(Mrad) \
        + (0.019993 - 0.000101 * T) * math.sin(2 * Mrad) \
        + 0.000289 * math.sin(3 * Mrad)

    # True longitude (deg)
    true_long = L0 + C

    # Obliquity of ecliptic (deg)
    eps0 = 23.439291 - 0.0130042 * T  # good enough here
    eps = math.radians(eps0)

    lam = math.radians(true_long)

    # Declination (rad)
    dec = math.asin(math.sin(eps) * math.sin(lam))
    subsolar_lat = math.degrees(dec)

    # Right Ascension (rad)
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    ra_deg = (math.degrees(ra) + 360.0) % 360.0

    # Greenwich Mean Sidereal Time (deg)
    # GMST approximation (deg)
    GMST = (280.46061837
            + 360.98564736629 * (JD - 2451545.0)
            + 0.000387933 * T * T
            - (T * T * T) / 38710000.0) % 360.0

    # Greenwich Hour Angle (deg) of the Sun
    GHA = (GMST - ra_deg) % 360.0

    # Subsolar longitude: where local hour angle = 0
    # lon_east = 180 - GHA  (then normalize to [-180, 180])
    subsolar_lon = (180.0 - GHA)
    subsolar_lon = (subsolar_lon + 180.0) % 360.0 - 180.0

    return subsolar_lat, subsolar_lon


def compute_day_night_index(dates, time_key):
    if not dates:
        return 0.0

    # 1) Вытаскиваем и чистим данные
    data = np.asarray(dates, dtype=float)  # если тут падает/делает nan — значит вход “грязный”
    if data.ndim != 2 or data.shape[1] < 3:
        return 0.0

    lat = data[:, 0]
    lon = data[:, 1]
    values = data[:, 2]

    # Убираем NaN/inf на входе (иначе расстояния станут nan)
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(values)
    if not np.any(valid):
        return 0.0

    lat = lat[valid]
    lon = lon[valid]
    values = values[valid]

    # 2) Нормализация “силы сигнала”
    # ROTI: обычно >=0. dTEC вариации: часто есть отрицательные.
    # Если есть отрицательные — берём abs, чтобы “волна” не самоуничтожалась.
    if np.nanmin(values) < 0:
        activity = np.abs(values)
    else:
        activity = values

    # 3) Подсолнечная точка по времени
    sub_lat, sub_lon = _subsolar_point(time_key)

    # 4) Расстояния до подсолнечной точки
    distances = great_circle_distance_vec(lat, lon, lat0=sub_lat, lon0=sub_lon)

    # На всякий: если где-то всё же nan пролез — выкинем
    ok = np.isfinite(distances) & np.isfinite(activity)
    if not np.any(ok):
        return 0.0
    distances = distances[ok]
    activity = activity[ok]

    # 5) День/ночь (граница: 90° от подсолнечной точки)
    day_limit = (math.pi / 2) * RE_meters
    day_mask = distances < day_limit
    night_mask = ~day_mask

    # 6) Веса дня: ближе к подсолнечной точке — больше
    # Нормируем так, чтобы на границе дня вес был 0, в центре — 1.
    total_day = 0.0
    if np.any(day_mask):
        d_day = distances[day_mask]
        a_day = activity[day_mask]
        w_day = 1.0 - (d_day / day_limit)  # от 1 до 0
        w_day = np.clip(w_day, 0.0, 1.0)
        total_day = float(np.sum(w_day * a_day))

    # 7) Ночь: делаем “слабый фон”, чтобы ночь почти не влияла
    # Можно считать суммой, но часто лучше средним (менее зависит от числа точек).
    night_gain = 0.15  # чем меньше, тем меньше влияние ночи
    total_night = 0.0
    if np.any(night_mask):
        a_night = activity[night_mask]
        total_night = float(np.mean(a_night))  # фон
        total_night *= night_gain

    # 8) Итоговый индекс
    eps = 1e-12
    if total_night <= eps:
        return float(total_day)

    return float(math.log(total_day / (total_night + eps)))
