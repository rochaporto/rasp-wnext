"""Synthetic WeatherNext-3-shaped Zarr store (ISA-like atmosphere) for a WRF smoke test."""
import sys
import numpy as np, pandas as pd, xarray as xr
from raspwn.wnext.source import LEVELS_HPA

init = np.datetime64("2026-10-01T00", "ns")
lat25 = np.arange(30.0, 50.01, 0.25); lon25 = np.arange(340.0, 360.0, 0.25)
lat10 = np.arange(30.0, 50.01, 0.1);  lon10 = np.arange(340.0, 360.0, 0.1)
lon25 = np.concatenate([lon25, np.arange(0, 5.01, 0.25)])
lon10 = np.concatenate([lon10, np.arange(0, 5.01, 0.1)])
lead = pd.to_timedelta(np.arange(0, 24, 6), unit="h"); sub = pd.to_timedelta(np.arange(6), unit="h")
d25 = ("init_time", "sample", "lead_time", "lead_subtime", "lat_0p25", "lon_0p25")
d10 = ("init_time", "sample", "lead_time", "lead_subtime", "lat_0p1", "lon_0p1")
s25 = (1, 2, lead.size, sub.size, lat25.size, lon25.size)
s10 = (1, 2, lead.size, sub.size, lat10.size, lon10.size)
LAT = lat25[None, None, None, None, :, None]
p = np.array(LEVELS_HPA, float)
z = 44330.8 * (1 - (p / 1013.25) ** 0.190263)
t = np.maximum(288.15 - 0.0065 * z, 216.65)
v = {}
for k, lev in enumerate(LEVELS_HPA):
    one = np.ones(s25, "f4")
    tk = t[k] + 3.0 - 0.5 * (LAT - 40)                 # warmer south
    v[f"temperature_{lev}"] = (d25, one * tk)
    v[f"geopotential_{lev}"] = (d25, one * (z[k] - 8.0 * (LAT - 40) * (lev < 900)) * 9.80665)
    rh = 0.6 if lev >= 700 else 0.3
    es = 611.2 * np.exp(17.67 * (tk - 273.15) / (tk - 29.65))
    v[f"specific_humidity_{lev}"] = (d25, one * (0.622 * rh * es / (lev * 100.0)))
    v[f"u_component_of_wind_{lev}"] = (d25, one * (3.0 + 0.02 * (1000 - lev)))
    v[f"v_component_of_wind_{lev}"] = (d25, one * 1.0)
    v[f"vertical_velocity_{lev}"] = (d25, one * 0.0)
o10 = np.ones(s10, "f4"); LAT10 = lat10[None, None, None, None, :, None]
v["temperature_2m"] = (d10, o10 * (291.0 - 0.5 * (LAT10 - 40)))
v["dewpoint_temperature_2m"] = (d10, o10 * (282.0 - 0.5 * (LAT10 - 40)))
v["u_component_of_wind_10m"] = (d10, o10 * 2.5)
v["v_component_of_wind_10m"] = (d10, o10 * 0.8)
v["mean_sea_level_pressure"] = (d10, o10 * 101500.0)
v["sea_surface_temperature"] = (d10, o10 * 292.0)
ds = xr.Dataset(v, coords=dict(init_time=[init], sample=[0, 1], lead_time=lead, lead_subtime=sub,
                               lat_0p25=lat25, lon_0p25=lon25, lat_0p1=lat10, lon_0p1=lon10))
ds.to_zarr(sys.argv[1], mode="w")
print(ds)
