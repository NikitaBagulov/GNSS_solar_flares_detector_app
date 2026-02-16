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

    cosgamma = np.clip(cosgamma, -1.0, 1.0)
    return RE_meters * np.arccos(cosgamma)


def _to_utc_datetime(dt: datetime.datetime) -> datetime.datetime:
    if dt.tzinfo is None:
        return dt
    return dt.astimezone(datetime.timezone.utc).replace(tzinfo=None)


def _subsolar_point(dt_value):
    year, month = dt_value.year, dt_value.month
    day = dt_value.day + (dt_value.hour + (dt_value.minute + dt_value.second / 60.0) / 60.0) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + (A // 4)
    jd = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5

    T = (jd - 2451545.0) / 36525.0
    L0 = (280.46646 + 36000.76983 * T + 0.0003032 * T * T) % 360.0
    M = (357.52911 + 35999.05029 * T - 0.0001537 * T * T) % 360.0

    Mrad = math.radians(M)
    C = (
        (1.914602 - 0.004817 * T - 0.000014 * T * T) * math.sin(Mrad)
        + (0.019993 - 0.000101 * T) * math.sin(2 * Mrad)
        + 0.000289 * math.sin(3 * Mrad)
    )
    true_long = L0 + C

    eps0 = 23.439291 - 0.0130042 * T
    eps = math.radians(eps0)
    lam = math.radians(true_long)

    dec = math.asin(math.sin(eps) * math.sin(lam))
    subsolar_lat = math.degrees(dec)

    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    ra_deg = (math.degrees(ra) + 360.0) % 360.0

    gmst = (
        280.46061837
        + 360.98564736629 * (jd - 2451545.0)
        + 0.000387933 * T * T
        - (T * T * T) / 38710000.0
    ) % 360.0

    gha = (gmst - ra_deg) % 360.0
    subsolar_lon = -gha
    subsolar_lon = (subsolar_lon + 180.0) % 360.0 - 180.0

    return subsolar_lat, subsolar_lon


def _weighted_percentile(x, w, q):
    x = np.asarray(x, dtype=float)
    w = np.asarray(w, dtype=float)

    m = np.isfinite(x) & np.isfinite(w) & (w > 0)
    if not np.any(m):
        return np.nan

    x = x[m]
    w = w[m]

    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]

    cdf = np.cumsum(w)
    cdf /= cdf[-1]

    return float(np.interp(q / 100.0, cdf, x))


def compute_day_night_index(dates, time_key):
    if not dates:
        return 0.0

    data = np.asarray(dates, dtype=float)
    if data.ndim != 2 or data.shape[1] < 3:
        return 0.0

    lat = data[:, 0]
    lon = data[:, 1]
    values = data[:, 2]

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(values)
    if not np.any(valid):
        return 0.0

    lat = lat[valid]
    lon = lon[valid]
    values = values[valid]

    activity = np.abs(values) if np.nanmin(values) < 0 else values

    sub_lat, sub_lon = _subsolar_point(time_key)
    distances = great_circle_distance_vec(lat, lon, lat0=sub_lat, lon0=sub_lon)

    ok = np.isfinite(distances) & np.isfinite(activity)
    if not np.any(ok):
        return 0.0

    distances = distances[ok]
    activity = activity[ok]

    chi = distances / RE_meters
    cos_chi = np.cos(chi)

    day_mask = chi < (math.pi / 2)
    night_mask = ~day_mask

    eps = 1e-12

    # ---- DAY: взвешенное СРЕДНЕЕ (устойчиво к числу точек) ----
    day_mean = 0.0
    if np.any(day_mask):
        a = activity[day_mask]
        mu = np.clip(cos_chi[day_mask], 0.0, 1.0)

        # мягкий вес по освещённости (не душим терминатор как w_geom)
        w_day = mu ** 0.5  # можно 1.0, если хочешь ровно cos(chi)

        wsum = float(np.sum(w_day))
        if wsum > 0.0:
            day_mean = float(np.sum(w_day * a) / (wsum + eps))

    # ---- NIGHT: обычное СРЕДНЕЕ (без gain) ----
    night_mean = 0.0
    if np.any(night_mask):
        night_mean = float(np.mean(activity[night_mask]))

    # если ночи нет — вернём просто дневной уровень
    if night_mean <= eps:
        return float(max(day_mean, 0.0))

    # ---- Индекс режима "2": насколько день выше ночного среднего, всегда >= 0 ----
    excess = (day_mean - night_mean) / (night_mean + eps)  # относительное превышение
    excess = max(excess, 0.0)
    return float(math.log1p(excess))
