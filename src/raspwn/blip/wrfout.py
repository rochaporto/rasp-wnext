"""Compute soaring parameters from a WRF ``wrfout`` file."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from .. import thermo
from . import params as bp

# name -> (units, description). Order is the display order.
PARAMETERS: dict[str, tuple[str, str]] = {
    "wstar": ("m/s", "Thermal updraft velocity W*"),
    "bsratio": ("", "Buoyancy/shear ratio"),
    "hwcrit": ("m", "Height of critical updraft (225 fpm), MSL"),
    "dwcrit": ("m", "Depth of critical updraft (225 fpm), AGL"),
    "hglider": ("m", "Thermalling height, MSL"),
    "dglider": ("m", "Thermalling height, AGL"),
    "hbl": ("m", "Boundary layer top, MSL"),
    "dbl": ("m", "Boundary layer depth"),
    "zsfclcl": ("m", "Cumulus cloudbase (surface parcel LCL), MSL"),
    "zsfclcldif": ("m", "Cumulus potential (BL top - cloudbase)"),
    "zblcl": ("m", "Overdevelopment cloudbase, MSL"),
    "zblcldif": ("m", "Overdevelopment potential (BL top - OD base)"),
    "stars": ("", "Soaring rating (0-5)"),
    "pfd": ("km/h", "Cross-country speed this hour (sums to potential flight distance)"),
    "sfcsun": ("W/m2", "Surface solar radiation"),
    "sfcshf": ("W/m2", "Surface sensible heat flux"),
    "sfctemp": ("C", "Surface temperature (2 m)"),
    "sfcdewpt": ("C", "Surface dew point (2 m)"),
    "sfcwind": ("m/s", "Surface wind (10 m)"),
    "blwind": ("m/s", "Boundary-layer averaged wind"),
    "bltopwind": ("m/s", "Wind at BL top"),
    "blwindshear": ("m/s", "BL vertical wind shear"),
    "blcloud": ("%", "Max cloud fraction within the BL"),
    "cfracl": ("%", "Low cloud cover"),
    "cfracm": ("%", "Mid cloud cover"),
    "cfrach": ("%", "High cloud cover"),
    "rain1": ("mm", "Precipitation since previous output"),
}

# Aggregated over all output times of a day
DAILY_PARAMETERS: dict[str, tuple[str, str]] = {
    "pfd_tot": ("km", "Potential flight distance (sum of hourly pfd)"),
}

# Vector parameters: name -> (u, v) of the plotted magnitude
VECTORS = {"sfcwind": ("u10", "v10"), "blwind": ("ublavg", "vblavg"), "bltopwind": ("ubltop", "vbltop")}


@dataclass
class Grid:
    lat: np.ndarray  # (ny, nx)
    lon: np.ndarray
    attrs: dict


def _destagger(a: np.ndarray, axis: int) -> np.ndarray:
    n = a.shape[axis]
    return 0.5 * (np.take(a, range(n - 1), axis=axis) + np.take(a, range(1, n), axis=axis))


def read_state(path: str | Path, time_index: int = 0) -> tuple[dict, Grid, datetime]:
    from netCDF4 import Dataset

    with Dataset(path) as nc:
        def v(name):
            return np.asarray(nc.variables[name][time_index], dtype=np.float64)

        times = nc.variables["Times"][time_index].tobytes().decode()
        valid = datetime.strptime(times, "%Y-%m-%d_%H:%M:%S")
        s = {
            "ter": v("HGT"),
            "pblh": v("PBLH"),
            "hfx": v("HFX"),
            "lh": v("LH"),
            "t2": v("T2"),
            "q2": v("Q2"),
            "psfc": v("PSFC"),
            "u10": v("U10"),
            "v10": v("V10"),
            "swdown": v("SWDOWN"),
            "rain": v("RAINC") + v("RAINNC"),
            "zfull": (v("PH") + v("PHB")) / thermo.G,
            "p": v("P") + v("PB"),
            "theta": v("T") + 300.0,
            "qv": v("QVAPOR"),
            "u": _destagger(v("U"), 2),
            "v": _destagger(v("V"), 1),
            "cldfra": v("CLDFRA") if "CLDFRA" in nc.variables else None,
            "cosalpha": v("COSALPHA"),
            "sinalpha": v("SINALPHA"),
        }
        grid = Grid(v("XLAT"), v("XLONG"), {a: nc.getncattr(a) for a in nc.ncattrs()})
    # grid-relative -> earth-relative winds
    ca, sa = s["cosalpha"], s["sinalpha"]
    s["u"], s["v"] = s["u"] * ca - s["v"] * sa, s["v"] * ca + s["u"] * sa
    s["u10"], s["v10"] = s["u10"] * ca - s["v10"] * sa, s["v10"] * ca + s["u10"] * sa
    s["z"] = _destagger(s["zfull"], 0)
    s["t"] = s["theta"] * (s["p"] / 1.0e5) ** thermo.KAPPA
    return s, grid, valid


def compute(s: dict, prev_rain: np.ndarray | None = None) -> dict[str, np.ndarray]:
    ter, pblh = s["ter"], s["pblh"]
    zf_agl = s["zfull"] - ter
    z_agl = s["z"] - ter
    out: dict[str, np.ndarray] = {}

    ws = bp.wstar(s["hfx"], s["lh"], s["t2"], s["psfc"], pblh)
    out["wstar"] = ws
    out["hbl"] = ter + pblh
    out["dbl"] = pblh
    out["hwcrit"] = bp.hcrit(ws, ter, pblh)
    out["dwcrit"] = out["hwcrit"] - ter

    out["ublavg"] = bp.bl_average(s["u"], zf_agl, pblh)
    out["vblavg"] = bp.bl_average(s["v"], zf_agl, pblh)
    out["blwind"] = np.hypot(out["ublavg"], out["vblavg"])
    out["ubltop"] = bp.value_at(s["u"], z_agl, pblh)
    out["vbltop"] = bp.value_at(s["v"], z_agl, pblh)
    out["bltopwind"] = np.hypot(out["ubltop"], out["vbltop"])
    out["blwindshear"] = np.hypot(out["ubltop"] - s["u"][0], out["vbltop"] - s["v"][0])
    out["bsratio"] = bp.bsratio(ws, out["blwind"])

    out["zsfclcl"] = bp.sfc_lcl(s["t2"], s["q2"], s["psfc"], s["z"], s["p"])
    out["zsfclcldif"] = ter + pblh - out["zsfclcl"]
    zblcl = bp.bl_lcl(s["qv"], zf_agl, s["z"], s["t"], s["p"], pblh)
    out["zblcl"] = np.where(np.isfinite(zblcl), zblcl, np.nan)
    out["zblcldif"] = np.where(np.isfinite(zblcl), ter + pblh - zblcl, -pblh)

    hgl = np.minimum(out["hwcrit"], out["zsfclcl"])
    hgl = np.minimum(hgl, np.where(np.isfinite(zblcl), zblcl, np.inf))
    out["hglider"] = np.maximum(hgl, ter)
    out["dglider"] = out["hglider"] - ter

    out["stars"] = bp.stars(out["dglider"], out["zsfclcldif"], out["blwind"], out["bsratio"])
    out["pfd"] = bp.pfd_speed(ws, out["dglider"], out["zsfclcldif"], out["bsratio"])

    out["sfcsun"] = s["swdown"]
    out["sfcshf"] = s["hfx"]
    out["sfctemp"] = s["t2"] - 273.15
    out["sfcdewpt"] = thermo.dewpoint_from_mixing_ratio(s["q2"], s["psfc"]) - 273.15
    out["sfcwind"] = np.hypot(s["u10"], s["v10"])
    out["u10"], out["v10"] = s["u10"], s["v10"]

    if s["cldfra"] is not None:
        out["blcloud"] = 100.0 * np.maximum(bp.bl_max(s["cldfra"], z_agl, pblh), 0.0)
        for k, val in bp.cloud_layers(s["cldfra"], s["p"]).items():
            out[k] = 100.0 * val
    out["rain1"] = np.maximum(s["rain"] - (0.0 if prev_rain is None else prev_rain), 0.0)
    return out
