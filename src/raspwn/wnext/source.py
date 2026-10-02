"""Read Google WeatherNext forecasts and normalise them.

WeatherNext 3 stores its full 64-member ensemble as Zarr v3 on GCS
(requester pays). Layout, per the public docs:

    gs://weathernext3_spatial/weathernext_3_0_0/zarr/
        2025/6hr_inits.zarr                     (one store, init_time dim)
        2026_to_present/<YYYYMMDD_HHhr_XX_preds>/predictions.zarr

Pressure-level fields (13 levels, 0.25 deg) only exist for the 6-hourly
inits; surface fields are on a 0.1 deg grid. Lead times are 6-hourly with
an extra ``lead_subtime`` offset giving hourly steps.

Whatever the raw layout, :func:`load` returns two datasets with a
canonical, model-agnostic shape that the rest of the pipeline relies on:

* ``upper``:   t, z, q, u, v, w   dims (time, level[hPa], lat, lon)
* ``surface``: t2m, d2m, u10, v10, msl, [sst]   dims (time, lat, lon)

lat ascending, lon in [-180, 180) ascending, time = valid time (UTC).
All raw names/dims are configurable (``WeatherNextConfig``) because the
store schema is not formally versioned; run ``raspwn inspect`` to check.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import xarray as xr

log = logging.getLogger(__name__)

UPPER_VARS = {
    "t": "temperature",
    "z": "geopotential",
    "q": "specific_humidity",
    "u": "u_component_of_wind",
    "v": "v_component_of_wind",
    "w": "vertical_velocity",
}

SURFACE_VARS = {
    "t2m": "temperature_2m",
    "d2m": "dewpoint_temperature_2m",
    "u10": "u_component_of_wind_10m",
    "v10": "v_component_of_wind_10m",
    "msl": "mean_sea_level_pressure",
    "sst": "sea_surface_temperature",
}
OPTIONAL_SURFACE = {"sst"}

LEVELS_HPA = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]


@dataclass
class WeatherNextConfig:
    bucket: str = "weathernext3_spatial"
    root: str = "weathernext_3_0_0/zarr"
    billing_project: str | None = None
    # Open this instead of GCS (local Zarr copy, or any URL xarray can open).
    path: str | None = None
    # int = one ensemble member, "mean" = ensemble mean
    member: int | str = 0
    member_dim: str = "sample"
    init_dim: str = "init_time"
    lead_dim: str = "lead_time"
    lead_subdim: str = "lead_subtime"
    levels: list[int] = field(default_factory=lambda: list(LEVELS_HPA))
    upper_vars: dict[str, str] = field(default_factory=lambda: dict(UPPER_VARS))
    surface_vars: dict[str, str] = field(default_factory=lambda: dict(SURFACE_VARS))


# --------------------------------------------------------------------------
# Opening the store
# --------------------------------------------------------------------------


def _gcs_store(cfg: WeatherNextConfig, prefix: str):
    import obstore
    import zarr

    opts = {}
    if cfg.billing_project:
        opts["default_headers"] = {"x-goog-user-project": cfg.billing_project}
    gcs = obstore.store.GCSStore(bucket=cfg.bucket, prefix=prefix, client_options=opts)
    return zarr.storage.ObjectStore(gcs, read_only=True)


def _list_init_dirs(cfg: WeatherNextConfig) -> list[str]:
    import obstore

    opts = {}
    if cfg.billing_project:
        opts["default_headers"] = {"x-goog-user-project": cfg.billing_project}
    gcs = obstore.store.GCSStore(bucket=cfg.bucket, client_options=opts)
    res = obstore.list_with_delimiter(gcs, f"{cfg.root}/2026_to_present/")
    return sorted(p.rstrip("/").rsplit("/", 1)[-1] for p in res["common_prefixes"])


def latest_init(cfg: WeatherNextConfig) -> datetime:
    """Most recent 6-hourly init (the only ones carrying pressure levels)."""
    for name in reversed(_list_init_dirs(cfg)):
        try:
            dt = datetime.strptime(name[:13], "%Y%m%d_%Hhr")
        except ValueError:
            continue
        if dt.hour % 6 == 0:
            return dt
    raise RuntimeError("no WeatherNext 6-hourly init found")


def open_raw(cfg: WeatherNextConfig, init: datetime) -> xr.Dataset:
    """Open the raw Zarr store holding ``init`` (lazy, no data read)."""
    if cfg.path:
        return xr.open_zarr(cfg.path, chunks={})
    if init.year >= 2026:
        stamp = init.strftime("%Y%m%d_%Hhr_")
        dirs = [d for d in _list_init_dirs(cfg) if d.startswith(stamp)]
        if not dirs:
            raise FileNotFoundError(f"no WeatherNext run for {init:%Y-%m-%d %HZ}")
        prefix = f"{cfg.root}/2026_to_present/{dirs[-1]}/predictions.zarr"
    else:
        prefix = f"{cfg.root}/{init.year}/6hr_inits.zarr"
    log.info("opening gs://%s/%s", cfg.bucket, prefix)
    return xr.open_zarr(_gcs_store(cfg, prefix), chunks={})


# --------------------------------------------------------------------------
# Normalisation helpers
# --------------------------------------------------------------------------


def _horizontal_dims(da: xr.DataArray) -> tuple[str, str]:
    lat = [d for d in da.dims if d.startswith("lat")]
    lon = [d for d in da.dims if d.startswith("lon")]
    if len(lat) != 1 or len(lon) != 1:
        raise ValueError(f"{da.name}: cannot find lat/lon dims in {da.dims}")
    return lat[0], lon[0]


def _subset(da: xr.DataArray, bbox: tuple[float, float, float, float]) -> xr.DataArray:
    """Cut a (west, south, east, north) box; output lon in [-180, 180)."""
    latd, lond = _horizontal_dims(da)
    west, south, east, north = bbox
    lat = da[latd].values
    lon180 = (da[lond].values + 180.0) % 360.0 - 180.0

    ilat = np.where((lat >= south) & (lat <= north))[0]
    ilat = ilat[np.argsort(lat[ilat])]
    ilon = np.where((lon180 >= west) & (lon180 <= east))[0]
    ilon = ilon[np.argsort(lon180[ilon])]
    if ilat.size < 2 or ilon.size < 2:
        raise ValueError(f"{da.name}: bbox {bbox} selects no grid points")
    out = da.isel({latd: ilat, lond: ilon})
    out = out.assign_coords({latd: lat[ilat], lond: lon180[ilon]})
    return out.rename({latd: "lat", lond: "lon"})


def _as_timedelta(values) -> np.ndarray:
    values = np.asarray(values)
    if np.issubdtype(values.dtype, np.timedelta64):
        return values.astype("timedelta64[s]")
    return (values.astype("float64") * 3600.0).astype("timedelta64[s]")  # hours


def _flatten_time(
    da: xr.DataArray, cfg: WeatherNextConfig, init: datetime, start: datetime, end: datetime
) -> xr.DataArray:
    """(init, lead[, subtime]) -> valid ``time``, restricted to [start, end]."""
    if cfg.init_dim in da.dims:
        inits = da[cfg.init_dim].values.astype("datetime64[s]")
        idx = np.where(inits == np.datetime64(init, "s"))[0]
        if idx.size == 0:
            raise KeyError(f"init {init} not in store")
        da = da.isel({cfg.init_dim: int(idx[0])}, drop=True)
    elif cfg.init_dim in da.coords:
        da = da.drop_vars(cfg.init_dim)

    lead, sub = cfg.lead_dim, cfg.lead_subdim
    if lead in da.dims and sub in da.dims:
        da = da.stack(_t=(lead, sub))
        offs = _as_timedelta(da[lead].values) + _as_timedelta(da[sub].values)
        da = da.drop_vars(["_t", lead, sub]).assign_coords(_t=offs)
    elif lead in da.dims:
        da = da.rename({lead: "_t"}).assign_coords(_t=_as_timedelta(da[lead].values))
    else:
        raise ValueError(f"{da.name}: no lead time dim ({lead}) in {da.dims}")

    valid = np.datetime64(init, "s") + da["_t"].values
    keep = np.where((valid >= np.datetime64(start, "s")) & (valid <= np.datetime64(end, "s")))[0]
    keep = keep[np.argsort(valid[keep])]
    _, first = np.unique(valid[keep], return_index=True)
    keep = keep[first]
    out = da.isel(_t=keep).assign_coords(_t=valid[keep]).rename(_t="time")
    return out


def _select_member(da: xr.DataArray, cfg: WeatherNextConfig) -> xr.DataArray:
    if cfg.member_dim not in da.dims:
        return da
    if cfg.member == "mean":
        return da.mean(cfg.member_dim)
    return da.isel({cfg.member_dim: int(cfg.member)}, drop=True)


def _upper_var(ds: xr.Dataset, base: str, cfg: WeatherNextConfig) -> xr.DataArray:
    """Pressure-level variable as (…, level, lat, lon) whatever the layout."""
    if base in ds:
        da = ds[base]
        lev = [d for d in da.dims if "level" in d or d in ("isobaricInhPa", "plev")]
        if not lev:
            raise ValueError(f"{base} has no level dim: {da.dims}")
        da = da.rename({lev[0]: "level"})
        return da.sel(level=cfg.levels)
    slabs = []
    for lev in cfg.levels:
        for name in (f"{base}_{lev}", f"{lev}_{base}"):
            if name in ds:
                slabs.append(ds[name])
                break
        else:
            raise KeyError(f"missing {base} at {lev} hPa")
    return xr.concat(slabs, dim=xr.DataArray(cfg.levels, dims="level", name="level"))


def normalise(
    ds: xr.Dataset,
    cfg: WeatherNextConfig,
    init: datetime,
    start: datetime,
    end: datetime,
    bbox: tuple[float, float, float, float],
) -> tuple[xr.Dataset, xr.Dataset]:
    def prep(da: xr.DataArray) -> xr.DataArray:
        da = _select_member(da, cfg)
        da = _subset(da, bbox)
        return _flatten_time(da, cfg, init, start, end)

    upper = xr.Dataset(
        {k: prep(_upper_var(ds, v, cfg)) for k, v in cfg.upper_vars.items()}
    ).transpose("time", "level", "lat", "lon")

    sfc = {}
    for k, v in cfg.surface_vars.items():
        if v not in ds:
            if k in OPTIONAL_SURFACE:
                continue
            raise KeyError(f"missing surface variable {v}")
        sfc[k] = prep(ds[v])
    surface = xr.Dataset(sfc).transpose("time", "lat", "lon")

    # metgrid needs every field at every time; keep the common valid times
    times = np.intersect1d(upper.time.values, surface.time.values)
    if times.size == 0:
        raise ValueError("no common valid times between upper-air and surface data")
    return upper.sel(time=times), surface.sel(time=times)


def load(
    cfg: WeatherNextConfig,
    init: datetime,
    start: datetime,
    end: datetime,
    bbox: tuple[float, float, float, float],
    margin_deg: float = 2.0,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Fetch and normalise one forecast window; returns in-memory datasets."""
    west, south, east, north = bbox
    padded = (west - margin_deg, south - margin_deg, east + margin_deg, north + margin_deg)
    raw = open_raw(cfg, init)
    upper, surface = normalise(raw, cfg, init, start, end, padded)
    log.info("downloading %d valid times x %d levels", upper.sizes["time"], upper.sizes["level"])
    return upper.load(), surface.load()


def describe(cfg: WeatherNextConfig, init: datetime) -> str:
    """Human-readable schema dump, to adapt the config to the live store."""
    ds = open_raw(cfg, init)
    lines = [f"dims: {dict(ds.sizes)}", "coords:"]
    for c in ds.coords:
        v = ds[c].values
        lines.append(f"  {c}: {v.dtype} [{v.flat[0]} .. {v.flat[-1]}] n={v.size}")
    lines.append("data_vars:")
    for name, da in ds.data_vars.items():
        lines.append(f"  {name}{da.dims} {da.dtype} {da.attrs.get('units', '')}")
    return "\n".join(lines)


def window(init: datetime, start_offset_h: int, length_h: int) -> tuple[datetime, datetime]:
    start = init + timedelta(hours=start_offset_h)
    return start, start + timedelta(hours=length_h)
