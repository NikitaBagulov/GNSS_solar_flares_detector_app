import h5py
from datetime import datetime, UTC

f = h5py.File("C:\\Users\\Chuiko\\Documents\\GitHub\\GNSS_solar_flares_detector_app\\data\\2025-11-11\\simurg_hdf\\simurg_hdf_20251111.h5")
sites = list(f.keys())[:]
site = sites[0]
sats = list(f[site].keys())[:]
times = list()
print(len(sats))
for sat in sats:
    times.extend(f[site][sat]["timestamp"][:])
times = list(set(times))
times.sort()

start = datetime.fromtimestamp(times[0])
start = start.replace(tzinfo=UTC)
end = datetime.fromtimestamp(times[-1])
end = end.replace(tzinfo=UTC)
print(start, end)

start = datetime.fromtimestamp(times[0], UTC)
end = datetime.fromtimestamp(times[-1], UTC)
print(start, end)

t = datetime(2025, 11, 11)
t = t.replace(tzinfo=UTC)
if not t == datetime.fromtimestamp(times[0], UTC):
    print("Check times")



