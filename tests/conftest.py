from datetime import datetime

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from raspwn.wnext.source import LEVELS_HPA, SURFACE_VARS

INIT = datetime(2026, 10, 2, 0)


def std_atmosphere(p_hpa):
    """ISA temperature [K] and height [m] for pressure levels."""
    p = np.asarray(p_hpa, float)
    z = 44330.8 * (1 - (p / 1013.25) ** 0.190263)
    t = np.maximum(288.15 - 0.0065 * z, 216.65)
    return t, z


@pytest.fixture
def raw_wn3():
    """Small synthetic store shaped like WeatherNext 3 raw ensemble output.

    Pressure fields are flattened per level (``temperature_850``), surface
    fields sit on a finer grid, lon is 0..360, lat ascending, and lead
    time is split into 6-hourly ``lead_time`` + hourly ``lead_subtime``.
    """
    rng = np.random.default_rng(0)
    # coarse spacing keeps the fixture small; only the layout matters here
    lat25 = np.arange(30.0, 50.01, 2.0)
    lon25 = np.arange(0.0, 360.0, 2.0)
    lat10 = np.arange(30.0, 50.01, 1.0)
    lon10 = np.arange(0.0, 360.0, 1.0)
    sample = np.arange(2)
    lead = pd.to_timedelta(np.arange(0, 24, 6), unit="h")
    sub = pd.to_timedelta(np.arange(0, 6), unit="h")
    dims25 = ("init_time", "sample", "lead_time", "lead_subtime", "lat_0p25", "lon_0p25")
    dims10 = ("init_time", "sample", "lead_time", "lead_subtime", "lat_0p1", "lon_0p1")
    shape25 = (1, sample.size, lead.size, sub.size, lat25.size, lon25.size)
    shape10 = (1, sample.size, lead.size, sub.size, lat10.size, lon10.size)

    # value encodes the valid hour so time flattening can be checked
    hours = (np.arange(lead.size)[:, None] * 6 + np.arange(sub.size)[None, :]).astype("f4")
    hours = hours[None, None, :, :, None, None]
    member = sample.astype("f4")[None, :, None, None, None, None]

    t_isa, z_isa = std_atmosphere(LEVELS_HPA)
    v = {}
    for k, p in enumerate(LEVELS_HPA):
        base = np.ones(shape25, "f4")
        v[f"temperature_{p}"] = (dims25, base * t_isa[k] + 0.01 * hours + member)
        v[f"geopotential_{p}"] = (dims25, base * z_isa[k] * 9.80665)
        v[f"specific_humidity_{p}"] = (dims25, base * 0.005 * (p / 1000.0))
        v[f"u_component_of_wind_{p}"] = (dims25, base * 5.0)
        v[f"v_component_of_wind_{p}"] = (dims25, base * -3.0)
        v[f"vertical_velocity_{p}"] = (dims25, rng.normal(0, 0.1, shape25).astype("f4"))
    s10 = np.ones(shape10, "f4")
    v[SURFACE_VARS["t2m"]] = (dims10, s10 * 290.0 + hours)
    v[SURFACE_VARS["d2m"]] = (dims10, s10 * 280.0)
    v[SURFACE_VARS["u10"]] = (dims10, s10 * 2.0)
    v[SURFACE_VARS["v10"]] = (dims10, s10 * 1.0)
    v[SURFACE_VARS["msl"]] = (dims10, s10 * 101325.0)
    v[SURFACE_VARS["sst"]] = (dims10, s10 * 291.0)
    return xr.Dataset(
        v,
        coords=dict(
            init_time=[np.datetime64(INIT, "ns")],
            sample=sample,
            lead_time=lead,
            lead_subtime=sub,
            lat_0p25=lat25,
            lon_0p25=lon25,
            lat_0p1=lat10,
            lon_0p1=lon10,
        ),
    )

