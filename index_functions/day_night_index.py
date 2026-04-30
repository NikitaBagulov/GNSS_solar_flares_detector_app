import numpy as np
import math
import datetime

RE_meters = 6371000.0


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
    dt_value = _to_utc_datetime(dt_value)

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


def _mad(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return 0.0
    med = np.median(x)
    return float(np.median(np.abs(x - med)))


def calculate_index(points, is_day=True):
    if len(points) == 0:
        return 0.0

    d = np.array([p[0] for p in points], dtype=float)
    values = np.array([p[1] for p in points], dtype=float)

    if is_day:
        weights = np.maximum(0.0, 1.0 - d / (2 * np.pi * RE_meters / 4.0))
        I = weights * values
    else:
         I = np.mean(values) #* np.mean(np.maximum(0.0, 1.0 - d / (2 * np.pi * RE_meters / 4.0)))

    I = np.round(I, 10)
    I = np.nan_to_num(I, nan=0.0)
    return np.sum(I)

def compute_day_night_index(
        points,
        time_key,
        debug=False,
        log_file="day_night_debug_log.csv",
        eps_abs=1e-6):

    data = np.asarray(points, dtype=float)
    if data.size == 0 or data.ndim != 2 or data.shape[1] < 3:
        return 0.0

    lat = data[:, 0]
    lon = data[:, 1]
    vals = data[:, 2]

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(vals)
    if not np.any(valid):
        return 0.0

    lat = lat[valid]
    lon = lon[valid]
    vals = vals[valid]

    sub_lat, sub_lon = _subsolar_point(time_key)

    distances = great_circle_distance_vec(lat, lon, lat0=sub_lat, lon0=sub_lon)

    gamma = distances / RE_meters
    delta = (np.pi / 2.0) - gamma

    day_mask = delta > 0
    night_mask = ~day_mask

    if np.sum(day_mask) == 0 or np.sum(night_mask) == 0:
        return 0.0

    d_term = np.abs(delta) * RE_meters

    d_day = d_term[day_mask]
    v_day = vals[day_mask]
    v_night = vals[night_mask]

    quarter_circ = 2 * np.pi * RE_meters / 4.0
    w_day = np.maximum(0.0, 1.0 - d_day / quarter_circ)

    if np.sum(w_day) > 0:
        mu_day = np.sum(v_day * w_day) / np.sum(w_day)
    else:
        mu_day = 0.0

    mu_night = np.median(v_night)

    num = mu_day - mu_night
    den = abs(mu_day) + abs(mu_night) + 0.05
    I_B = num / den

    # ---- DEBUG ----
    if debug:

        N_day = int(np.sum(day_mask))
        N_night = int(np.sum(night_mask))

        # запись в файл
        with open(log_file, "a") as f:
            f.write(
                f"{time_key},"
                f"{sub_lat:.6f},{sub_lon:.6f},"
                f"{N_day},{N_night},"
                f"{mu_day:.6f},{mu_night:.6f},"
                f"{num:.6f},{den:.6f},{I_B:.6f}\n"
            )

    return float(I_B)
