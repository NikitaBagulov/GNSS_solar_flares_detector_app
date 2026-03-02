import math
import numpy as np

# ----------------------------------------
# Реальное положение Солнца без библиотек
# ----------------------------------------

def solar_params(timestamp):
    y = timestamp.year
    m = timestamp.month
    d = timestamp.day
    hh = timestamp.hour
    mm = timestamp.minute
    ss = timestamp.second + timestamp.microsecond / 1_000_000

    jd = julian_day(y, m, d, hh, mm, ss)
    T = (jd - 2451545.0) / 36525.0

    decl = sun_position(jd)

    L0 = math.radians((280.46646 + 36000.76983*T) % 360)
    M = math.radians(357.52911 + 35999.05029*T)
    e = 0.016708634 - 0.000042037*T
    y_ = math.tan(math.radians(23.439291 - 0.0130042*T) / 2)**2

    eq_time = 4 * math.degrees(
        y_*math.sin(2*L0)
        - 2*e*math.sin(M)
        + 4*e*y_*math.sin(M)*math.cos(2*L0)
        - 0.5*y_*y_*math.sin(4*L0)
        - 1.25*e*e*math.sin(2*M)
    )

    return decl, eq_time


def julian_day(year, month, day, hour, minute, second):
    if month <= 2:
        year -= 1
        month += 12
    A = year // 100
    B = 2 - A + A // 4
    JD = int(365.25 * (year + 4716)) \
         + int(30.6001 * (month + 1)) \
         + day + B - 1524.5
    JD += (hour + minute/60 + second/3600) / 24
    return JD

def sun_position(jd):
    T = (jd - 2451545.0) / 36525.0

    L0 = 280.46646 + 36000.76983*T + 0.0003032*T*T
    L0 %= 360.0

    M = 357.52911 + 35999.05029*T - 0.0001537*T*T
    M = math.radians(M)

    C = (1.914602 - 0.004817*T - 0.000014*T*T)*math.sin(M)
    C += (0.019993 - 0.000101*T)*math.sin(2*M)
    C += 0.000289*math.sin(3*M)

    true_long = math.radians(L0 + C)

    epsilon = math.radians(23.439291 - 0.0130042*T)

    decl = math.asin(math.sin(epsilon) * math.sin(true_long))
    return decl

def solar_zenith(lat_deg, lon_deg, timestamp):
    y = timestamp.year
    m = timestamp.month
    d = timestamp.day
    hh = timestamp.hour
    mm = timestamp.minute
    ss = timestamp.second + timestamp.microsecond / 1_000_000

    jd = julian_day(y, m, d, hh, mm, ss)
    T = (jd - 2451545.0) / 36525.0

    # солнечная деклинация
    decl = sun_position(jd)

    # equation of time (approx) — для H
    L0 = math.radians((280.46646 + 36000.76983*T) % 360)
    M = math.radians(357.52911 + 35999.05029*T)
    e = 0.016708634 - 0.000042037*T
    y_ = math.tan(math.radians(23.439291 - 0.0130042*T) / 2)**2

    eq_time = 4 * math.degrees(
        y_*math.sin(2*L0)
        - 2*e*math.sin(M)
        + 4*e*y_*math.sin(M)*math.cos(2*L0)
        - 0.5*y_*y_*math.sin(4*L0)
        - 1.25*e*e*math.sin(2*M)
    )

    # местное солнечное время (в минутах)
    time_offset = eq_time + 4 * lon_deg
    tst = hh*60 + mm + ss/60 + time_offset

    # часовой угол (в радианах)
    H = math.radians((tst / 4) - 180)
    
    lat = math.radians(lat_deg)

    cos_chi = math.sin(lat)*math.sin(decl) + math.cos(lat)*math.cos(decl)*math.cos(H)
    cos_chi = max(-1.0, min(1.0, cos_chi))

    chi_deg = math.degrees(math.acos(cos_chi))
    return chi_deg, cos_chi

# ----------------------------------------
# ISFAI (Xiong et al., 2013)
# ----------------------------------------

def compute_isfai_index(dates, time_key):
    if not dates:
        return 0.0

    data = np.array(dates, dtype=float)

    lat = np.radians(data[:, 0])
    lon = data[:, 1]
    vals = np.abs(data[:, 2])

    # 🔒 базовая фильтрация
    valid = (
        np.isfinite(lat) &
        np.isfinite(lon) &
        np.isfinite(vals)
    )

    if not np.any(valid):
        return 0.0

    lat = lat[valid]
    lon = lon[valid]
    vals = vals[valid]

    decl, eq_time = solar_params(time_key)

    hh = time_key.hour
    mm = time_key.minute
    ss = time_key.second + time_key.microsecond / 1_000_000

    tst = hh * 60 + mm + ss / 60 + eq_time + 4 * lon
    H = np.radians((tst / 4) - 180)

    cos_chi = (
        np.sin(lat) * np.sin(decl)
        + np.cos(lat) * np.cos(decl) * np.cos(H)
    )

    cos_chi = np.clip(cos_chi, -1.0, 1.0)

    # 🔒 ещё раз чистим
    valid = np.isfinite(cos_chi)
    cos_chi = cos_chi[valid]
    vals = vals[valid]

    if len(cos_chi) == 0:
        return 0.0

    cos_limit = np.cos(np.radians(70))
    mask = cos_chi >= cos_limit

    if not np.any(mask):
        return 0.0

    numerator = np.sum(vals[mask])
    denominator = np.sum(cos_chi[mask])

    if not np.isfinite(denominator) or abs(denominator) < 1e-12:
        return 0.0

    return float(numerator / denominator)


