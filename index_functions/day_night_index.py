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


def _day_geometry(distance_from_subsolar, min_cos=1e-6):
    """Return day weights and cos(chi) divisors for subsolar distances.

    The linear distance weight is 1 at the subsolar point and 0 at the
    terminator.  The divisor is clipped only for numerical safety; the weight
    simultaneously tends to zero as cos(chi) tends to zero.
    """
    distances = np.asarray(distance_from_subsolar, dtype=float)
    quarter_circ = np.pi * RE_meters / 2.0
    weights = np.clip(1.0 - distances / quarter_circ, 0.0, 1.0)
    cos_chi = np.cos(distances / RE_meters)
    divisors = np.maximum(cos_chi, float(min_cos))
    return weights, divisors


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

DAY_NIGHT_VARIANTS = (
    "legacy",
    "distance_weight",
    "distance_weight_cos",
    "distance_weight_cos_sum",
)


def compute_day_night_components(
        points,
        time_key,
        variant="distance_weight_cos",
        eps_abs=1e-6,
        exclude_terminator_deg=0.0):
    """Return the day/night terms used by the normalized contrast index.

    Variants are explicit so comparisons never overwrite old results.
     legacy reproduces the historical reversed weight; distance_weight uses
     weight 1 at the subsolar point and 0 at the terminator; distance_weight_cos
     divides dayside values by cos(chi); and distance_weight_cos_sum uses the
     ISFAI-style aggregate denominator sum(w_i cos(chi_i)).
    """
    if variant not in DAY_NIGHT_VARIANTS:
        raise ValueError(f"Unknown day/night variant: {variant}")
    if eps_abs <= 0:
        raise ValueError("eps_abs must be positive")
    if not 0.0 <= exclude_terminator_deg < 90.0:
        raise ValueError("exclude_terminator_deg must be in [0, 90)")

    data = np.asarray(points, dtype=float)
    if data.size == 0 or data.ndim != 2 or data.shape[1] < 3:
        return None
    lat, lon, vals = data[:, 0], data[:, 1], data[:, 2]
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(vals)
    if not np.any(valid):
        return None
    lat, lon, vals = lat[valid], lon[valid], vals[valid]

    sub_lat, sub_lon = _subsolar_point(time_key)
    distances = great_circle_distance_vec(lat, lon, lat0=sub_lat, lon0=sub_lon)
    zenith_deg = np.degrees(distances / RE_meters)
    day_mask = zenith_deg < (90.0 - float(exclude_terminator_deg))
    night_mask = zenith_deg >= 90.0
    if not np.any(day_mask) or not np.any(night_mask):
        return None

    d_day = distances[day_mask]
    v_day = vals[day_mask]
    v_night = vals[night_mask]
    corrected_weights, cos_day = _day_geometry(d_day, min_cos=eps_abs)
    if variant == "legacy":
        distance_to_terminator = (np.pi / 2.0 * RE_meters) - d_day
        weights = np.clip(1.0 - distance_to_terminator / (np.pi / 2.0 * RE_meters), 0.0, 1.0)
        day_values = v_day
    elif variant == "distance_weight":
        weights = corrected_weights
        day_values = v_day
    elif variant == "distance_weight_cos":
        weights = corrected_weights
        day_values = v_day / cos_day
    elif variant == "distance_weight_cos_sum":
        weights = corrected_weights
        day_values = v_day
    else:
        raise ValueError(f"Unknown day/night variant: {variant}")

    if variant == "distance_weight_cos_sum":
        weight_sum = float(np.sum(weights * cos_day))
    else:
        weight_sum = float(np.sum(weights))
    if weight_sum <= 0:
        return None
    mu_day = float(np.sum(day_values * weights) / weight_sum)
    mu_night = float(np.median(v_night))
    numerator = mu_day - mu_night
    denominator = abs(mu_day) + abs(mu_night) + 0.05
    return {
        "variant": variant,
        "mu_day": mu_day,
        "mu_night": mu_night,
        "numerator": numerator,
        "denominator": denominator,
        "index": numerator / denominator,
        "n_day": int(np.sum(day_mask)),
        "n_night": int(np.sum(night_mask)),
        "subsolar_lat": float(sub_lat),
        "subsolar_lon": float(sub_lon),
    }


def compute_day_night_index(
        points,
        time_key,
        debug=False,
        log_file="day_night_debug_log.csv",
        eps_abs=1e-6,
        variant="distance_weight_cos",
        exclude_terminator_deg=0.0):
    components = compute_day_night_components(
        points, time_key, variant=variant, eps_abs=eps_abs,
        exclude_terminator_deg=exclude_terminator_deg,
    )
    if components is None:
        return 0.0
    if debug:
        with open(log_file, "a") as stream:
            stream.write(
                f"{time_key},{components['variant']},"
                f"{components['subsolar_lat']:.6f},{components['subsolar_lon']:.6f},"
                f"{components['n_day']},{components['n_night']},"
                f"{components['mu_day']:.6f},{components['mu_night']:.6f},"
                f"{components['numerator']:.6f},{components['denominator']:.6f},"
                f"{components['index']:.6f}\\n"
            )
    return float(components["index"])
