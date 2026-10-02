"""Turn normalised WeatherNext datasets into WPS intermediate files.

Produces one ``WNEXT:YYYY-MM-DD_HH`` file per valid time with the fields
metgrid needs for a pressure-level first guess:

  isobaric:  TT, UU, VV, RH, GHT          (13 levels, 0.25 deg)
  surface:   TT, UU, VV, RH (2 m / 10 m)  (0.1 deg)
             PMSL, SST, PSFC

WeatherNext has no land-surface state (soil, skin temperature, snow,
land mask, model terrain); those come from GFS as metgrid "constants"
(see :mod:`raspwn.wps.gfs`). PSFC is derived here by interpolating the
WeatherNext column to the GFS terrain height so that surface pressure is
consistent with the WeatherNext atmosphere rather than GFS's.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from .. import thermo
from .intermediate import LEVEL_MSL, LEVEL_SURFACE, Field, file_name, write_file

PREFIX = "WNEXT"


def _grid(da: xr.DataArray) -> dict:
    lat, lon = da["lat"].values, da["lon"].values
    dlat, dlon = np.diff(lat), np.diff(lon)
    if not (np.allclose(dlat, dlat[0], atol=1e-4) and np.allclose(dlon, dlon[0], atol=1e-4)):
        raise ValueError("grid is not regular lat/lon")
    return dict(
        startlat=float(lat[0]), startlon=float(lon[0]), dlat=float(dlat[0]), dlon=float(dlon[0])
    )


def bilinear(field2d: np.ndarray, src: dict, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Sample a regular-grid field (``src`` as in :func:`_grid`) at lat/lon points."""
    fy = (np.asarray(lat) - src["startlat"]) / src["dlat"]
    fx = ((np.asarray(lon) - src["startlon"]) % 360.0) / src["dlon"]
    ny, nx = field2d.shape
    fy = np.clip(fy, 0, ny - 1.000001)
    fx = np.clip(fx, 0, nx - 1.000001)
    y0, x0 = np.floor(fy).astype(int), np.floor(fx).astype(int)
    wy, wx = fy - y0, fx - x0
    f = field2d
    return (
        f[y0, x0] * (1 - wy) * (1 - wx)
        + f[y0 + 1, x0] * wy * (1 - wx)
        + f[y0, x0 + 1] * (1 - wy) * wx
        + f[y0 + 1, x0 + 1] * wy * wx
    )


def surface_pressure(
    ght: np.ndarray, levels_pa: np.ndarray, pmsl: np.ndarray, terrain: np.ndarray
) -> np.ndarray:
    """Surface pressure [Pa] by log-p interpolation of the column to ``terrain``.

    ght: (nlev, ny, nx) geopotential height [m]; pmsl, terrain: (ny, nx).
    The MSL pressure at z=0 is added to the column so low terrain below the
    1000 hPa surface is handled without extrapolation.
    """
    nlev, ny, nx = ght.shape
    lnp_lev = np.log(levels_pa)
    out = np.empty((ny, nx))
    for j in range(ny):
        for i in range(nx):
            z = np.concatenate(([0.0], ght[:, j, i]))
            lnp = np.concatenate(([np.log(pmsl[j, i])], lnp_lev))
            order = np.argsort(z)
            z, lnp = z[order], lnp[order]
            # keep the strictly increasing part (z=0 may sit above 1000 hPa)
            keep = np.concatenate(([True], np.diff(z) > 0))
            z, lnp = z[keep], lnp[keep]
            h = terrain[j, i]
            if h < z[0]:  # below the lowest point: linear extrapolation
                slope = (lnp[1] - lnp[0]) / (z[1] - z[0])
                out[j, i] = np.exp(lnp[0] + slope * (h - z[0]))
            else:
                out[j, i] = np.exp(np.interp(h, z, lnp))
    return out


def fields_for_time(
    upper: xr.Dataset,
    surface: xr.Dataset,
    t: np.datetime64,
    terrain: tuple[np.ndarray, dict] | None = None,
) -> list[Field]:
    """All intermediate fields for one valid time.

    ``terrain`` is (height[m], grid) of the land-surface source model; if
    given, PSFC is computed on the upper-air grid.
    """
    date = pd.Timestamp(t).to_pydatetime()
    up = upper.sel(time=t)
    sf = surface.sel(time=t)
    ug, sg = _grid(up.t), _grid(sf.t2m)
    out: list[Field] = []

    def add(name, units, desc, level, data, grid):
        out.append(Field(name, units, desc, float(level), date, data=np.asarray(data), **grid))

    levels_pa = up.level.values.astype(float) * 100.0
    ght = up.z.values / thermo.G
    for k, p in enumerate(levels_pa):
        t_k = up.t.values[k]
        add("TT", "K", "Temperature", p, t_k, ug)
        add("UU", "m s-1", "U", p, up.u.values[k], ug)
        add("VV", "m s-1", "V", p, up.v.values[k], ug)
        add("RH", "%", "Relative Humidity", p, thermo.rh_from_q(up.q.values[k], t_k, p), ug)
        add("GHT", "m", "Height", p, ght[k], ug)

    add("TT", "K", "Temperature at 2 m", LEVEL_SURFACE, sf.t2m.values, sg)
    add("UU", "m s-1", "U at 10 m", LEVEL_SURFACE, sf.u10.values, sg)
    add("VV", "m s-1", "V at 10 m", LEVEL_SURFACE, sf.v10.values, sg)
    add(
        "RH", "%", "Relative Humidity at 2 m", LEVEL_SURFACE,
        thermo.rh_from_dewpoint(sf.t2m.values, sf.d2m.values), sg,
    )
    add("PMSL", "Pa", "Sea-level Pressure", LEVEL_MSL, sf.msl.values, sg)
    if "sst" in sf:
        add("SST", "K", "Sea-Surface Temperature", LEVEL_SURFACE, sf.sst.values, sg)

    if terrain is not None:
        hgt, tgrid = terrain
        lat2d, lon2d = np.meshgrid(up.lat.values, up.lon.values, indexing="ij")
        hgt_u = bilinear(hgt, tgrid, lat2d, lon2d)
        pmsl_u = bilinear(sf.msl.values, sg, lat2d, lon2d)
        add("PSFC", "Pa", "Surface Pressure", LEVEL_SURFACE,
            surface_pressure(ght, levels_pa, pmsl_u, hgt_u), ug)
    return out


def write_intermediate(
    upper: xr.Dataset,
    surface: xr.Dataset,
    outdir: str | Path,
    terrain: tuple[np.ndarray, dict] | None = None,
    prefix: str = PREFIX,
) -> list[Path]:
    paths = []
    for t in upper.time.values:
        date: datetime = pd.Timestamp(t).to_pydatetime()
        fields = fields_for_time(upper, surface, t, terrain)
        paths.append(write_file(Path(outdir) / file_name(prefix, date), fields))
    return paths
