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


def _subsolar_point(dt: datetime.datetime):
    dt = _to_utc_datetime(dt)

    year, month = dt.year, dt.month
    day = dt.day + (dt.hour + (dt.minute + dt.second / 60.0) / 60.0) / 24.0
    if month <= 2:
        year -= 1
        month += 12

    A = year // 100
    B = 2 - A + (A // 4)
    JD = int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day + B - 1524.5

    T = (JD - 2451545.0) / 36525.0

    L0 = (280.46646 + 36000.76983 * T) % 360.0
    M = (357.52911 + 35999.05029 * T) % 360.0

    Mrad = math.radians(M)
    C = 1.914602 * math.sin(Mrad) + 0.019993 * math.sin(2 * Mrad)

    true_long = L0 + C
    eps = math.radians(23.439291)

    lam = math.radians(true_long)

    dec = math.asin(math.sin(eps) * math.sin(lam))
    subsolar_lat = math.degrees(dec)

    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    ra_deg = (math.degrees(ra) + 360.0) % 360.0

    GMST = (280.46061837
            + 360.98564736629 * (JD - 2451545.0)) % 360.0

    GHA = (GMST - ra_deg) % 360.0

    subsolar_lon = (180.0 - GHA)
    subsolar_lon = (subsolar_lon + 180.0) % 360.0 - 180.0

    return subsolar_lat, subsolar_lon


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

    if np.nanmin(values) < 0:
        activity = np.abs(values)
    else:
        activity = values

    sub_lat, sub_lon = _subsolar_point(time_key)

    distances = great_circle_distance_vec(lat, lon, lat0=sub_lat, lon0=sub_lon)

    ok = np.isfinite(distances) & np.isfinite(activity)
    if not np.any(ok):
        return 0.0

    distances = distances[ok]
    activity = activity[ok]

    # ---------- ВАЖНОЕ ИЗМЕНЕНИЕ НАЧИНАЕТСЯ ЗДЕСЬ ----------

    # зенитный угол Солнца (рад)
    chi = distances / RE_meters

    # косинус зенитного угла
    cos_chi = np.cos(chi)

    # день / ночь
    day_limit = math.pi / 2
    day_mask = chi < day_limit
    night_mask = ~day_mask

    total_day = 0.0
    if np.any(day_mask):
        d_day = distances[day_mask]
        a_day = activity[day_mask]
        mu = cos_chi[day_mask]

        # старый линейный вес
        w_geom = 1.0 - (d_day / (day_limit * RE_meters))
        w_geom = np.clip(w_geom, 0.0, 1.0)

        # НОВОЕ: физический вес cos(χ)
        w = w_geom * np.clip(mu, 0.0, 1.0)

        total_day = float(np.sum(w * a_day))

    # ночь — слабый фон
    night_gain = 0.15
    total_night = 0.0
    if np.any(night_mask):
        total_night = float(np.mean(activity[night_mask])) * night_gain

    eps = 1e-12
    if total_night <= eps:
        return float(total_day)

    return float(math.log(total_day / (total_night + eps)))
