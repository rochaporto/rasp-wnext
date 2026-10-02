"""Land-surface "constants" from GFS for the WRF initial state.

WeatherNext forecasts the atmosphere only. WRF's land surface model
still needs soil temperature/moisture, skin temperature, snow, sea ice,
a land mask and the source model's terrain at the initial time. We take
these from the GFS 0.25 deg analysis (f000) of the same cycle, pulled
(or the short-range forecast valid at the WRF start time) from the
NOAA open-data bucket on AWS with HTTP byte ranges (only the
~16 needed GRIB messages, a few MB), decode them with ecCodes, and write
a ``GFSLSM:YYYY-MM-DD_HH`` intermediate file used by metgrid through
``constants_name``.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import requests

from .intermediate import LEVEL_SURFACE, MISSING, Field, file_name, write_file

log = logging.getLogger(__name__)

PREFIX = "GFSLSM"
GFS_URL = (
    "https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
    "gfs.{d:%Y%m%d}/{d:%H}/atmos/gfs.t{d:%H}z.pgrb2.0p25.f{fh:03d}"
)

# (.idx pattern, WPS name, units, description, XLVL)
SOIL_LAYERS = [("0-0.1", "000010"), ("0.1-0.4", "010040"), ("0.4-1", "040100"), ("1-2", "100200")]
RECORDS: list[tuple[str, str, str, str, float]] = [
    (":LAND:surface:", "LANDSEA", "proprtn", "Land/Sea flag", LEVEL_SURFACE),
    (":ICEC:surface:", "SEAICE", "proprtn", "Ice flag", LEVEL_SURFACE),
    (":HGT:surface:", "SOILHGT", "m", "Terrain field of source analysis", LEVEL_SURFACE),
    (":TMP:surface:", "SKINTEMP", "K", "Skin temperature", LEVEL_SURFACE),
    (":WEASD:surface:", "SNOW", "kg m-2", "Water equivalent snow depth", LEVEL_SURFACE),
    (":SNOD:surface:", "SNOWH", "m", "Physical snow depth", LEVEL_SURFACE),
] + [
    rec
    for top_bot, tag in SOIL_LAYERS
    for rec in (
        (f":TSOIL:{top_bot} m below ground:", f"ST{tag}", "K", f"Soil T {tag}", LEVEL_SURFACE),
        (f":SOILW:{top_bot} m below ground:", f"SM{tag}", "fraction", f"Soil moist {tag}", LEVEL_SURFACE),
    )
]


def parse_idx(text: str) -> list[tuple[int, int | None, str]]:
    """GRIB .idx -> [(start, end_inclusive_or_None, line)]."""
    entries = []
    lines = [ln for ln in text.splitlines() if ln.strip()]
    offsets = [int(ln.split(":")[1]) for ln in lines]
    for k, ln in enumerate(lines):
        end = offsets[k + 1] - 1 if k + 1 < len(lines) else None
        entries.append((offsets[k], end, ln))
    return entries


def _decode(msg: bytes) -> tuple[np.ndarray, dict]:
    import eccodes

    h = eccodes.codes_new_from_message(msg)
    try:
        ni = eccodes.codes_get(h, "Ni")
        nj = eccodes.codes_get(h, "Nj")
        lat0 = eccodes.codes_get(h, "latitudeOfFirstGridPointInDegrees")
        lon0 = eccodes.codes_get(h, "longitudeOfFirstGridPointInDegrees")
        dlat = eccodes.codes_get(h, "jDirectionIncrementInDegrees")
        dlon = eccodes.codes_get(h, "iDirectionIncrementInDegrees")
        north_to_south = eccodes.codes_get(h, "jScansPositively") == 0
        missing = eccodes.codes_get(h, "missingValue")
        vals = eccodes.codes_get_values(h).reshape(nj, ni)
    finally:
        eccodes.codes_release(h)
    vals = np.where(vals == missing, np.nan, vals)
    if north_to_south:
        vals = vals[::-1]
        lat0 = lat0 - (nj - 1) * dlat
    return vals, dict(startlat=lat0, startlon=lon0, dlat=dlat, dlon=dlon)


def _crop(vals: np.ndarray, grid: dict, bbox) -> tuple[np.ndarray, dict]:
    """Cut a padded (west, south, east, north) box from a global grid."""
    west, south, east, north = bbox
    ny, nx = vals.shape
    j0 = max(int(np.floor((south - grid["startlat"]) / grid["dlat"])), 0)
    j1 = min(int(np.ceil((north - grid["startlat"]) / grid["dlat"])), ny - 1)
    i0 = int(np.floor((west - grid["startlon"]) / grid["dlon"]))
    i1 = int(np.ceil((east - grid["startlon"]) / grid["dlon"]))
    cols = np.arange(i0, i1 + 1) % nx  # handles boxes crossing the Greenwich meridian
    out = vals[j0 : j1 + 1][:, cols]
    return out, dict(
        startlat=grid["startlat"] + j0 * grid["dlat"],
        startlon=((grid["startlon"] + i0 * grid["dlon"]) + 180.0) % 360.0 - 180.0,
        dlat=grid["dlat"],
        dlon=grid["dlon"],
    )


def fetch(
    cycle: datetime, valid: datetime, bbox, session: requests.Session | None = None
) -> list[Field]:
    """Download and decode GFS land-surface fields of ``cycle`` valid at ``valid``."""
    s = session or requests.Session()
    fh = int((valid - cycle).total_seconds() // 3600)
    url = GFS_URL.format(d=cycle, fh=fh)
    r = s.get(url + ".idx", timeout=60)
    r.raise_for_status()
    index = parse_idx(r.text)

    west, south, east, north = bbox
    padded = (west - 2, south - 2, east + 2, north + 2)
    fields = []
    for pattern, name, units, desc, level in RECORDS:
        hits = [e for e in index if pattern in e[2]]
        if not hits:
            raise KeyError(f"GFS {cycle:%Y%m%d%H} f{fh:03d}: no record matching {pattern!r}")
        start, end, _ = hits[0]
        rng = f"bytes={start}-{'' if end is None else end}"
        resp = s.get(url, headers={"Range": rng}, timeout=120)
        resp.raise_for_status()
        vals, grid = _decode(resp.content)
        vals, grid = _crop(vals, grid, padded)
        fields.append(
            Field(name, units, desc, level, valid, data=vals, map_source="NCEP GFS", **grid)
        )
        log.debug("GFS %s ok (%s)", name, rng)
    return fields


def write(fields: list[Field], outdir: str | Path) -> Path:
    return write_file(Path(outdir) / file_name(PREFIX, fields[0].date), fields)


def terrain(fields: list[Field]) -> tuple[np.ndarray, dict]:
    f = next(f for f in fields if f.name == "SOILHGT")
    data = np.where(f.data <= MISSING / 10, np.nan, f.data)  # after a read_file()
    return data, dict(startlat=f.startlat, startlon=f.startlon, dlat=f.dlat, dlon=f.dlon)

