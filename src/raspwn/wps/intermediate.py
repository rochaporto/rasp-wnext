"""Reader/writer for the WPS intermediate file format (version 5).

This is what ungrib.exe produces and metgrid.exe consumes. Writing it
directly from Python lets us feed WRF with any data source (here:
WeatherNext) without going through GRIB and ungrib.

Each field is a sequence of big-endian Fortran unformatted records:

  1. IFV (int32, = 5)
  2. HDATE(c24) XFCST(f32) MAP_SOURCE(c32) FIELD(c9) UNITS(c25) DESC(c46)
     XLVL(f32) NX(i32) NY(i32) IPROJ(i32)
  3. projection info (only IPROJ=0, cylindrical equidistant, supported):
     STARTLOC(c8) STARTLAT STARTLON DELTALAT DELTALON EARTH_RADIUS (f32)
  4. IS_WIND_EARTH_REL (logical, int32)
  5. SLAB(NX, NY) float32, Fortran order (x varies fastest)

See WRF Users' Guide, chapter 3, "Writing Meteorological Data to the
Intermediate Format".
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

IFV = 5
EARTH_RADIUS_KM = 6367.47
MISSING = -1.0e30

# XLVL conventions used by metgrid
LEVEL_SURFACE = 200100.0
LEVEL_MSL = 201300.0


@dataclass
class Field:
    """One 2D slab on a regular lat/lon grid.

    ``data`` has shape (ny, nx) with row 0 at ``startlat`` (the south-west
    corner) and latitude increasing with the row index.
    """

    name: str
    units: str
    desc: str
    level: float
    date: datetime
    startlat: float
    startlon: float
    dlat: float
    dlon: float
    data: np.ndarray
    map_source: str = "WeatherNext"
    xfcst: float = 0.0
    earth_relative_wind: bool = True

    @property
    def ny(self) -> int:
        return self.data.shape[0]

    @property
    def nx(self) -> int:
        return self.data.shape[1]


def _record(f: BinaryIO, payload: bytes) -> None:
    marker = struct.pack(">i", len(payload))
    f.write(marker)
    f.write(payload)
    f.write(marker)


def _s(text: str, n: int) -> bytes:
    return text.encode("ascii")[:n].ljust(n, b" ")


def hdate(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H:%M:%S")


def write_field(f: BinaryIO, fld: Field) -> None:
    data = np.asarray(fld.data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"{fld.name}: expected 2D data, got shape {data.shape}")
    data = np.where(np.isfinite(data), data, MISSING).astype(">f4")
    ny, nx = data.shape

    _record(f, struct.pack(">i", IFV))
    _record(
        f,
        _s(hdate(fld.date), 24)
        + struct.pack(">f", fld.xfcst)
        + _s(fld.map_source, 32)
        + _s(fld.name, 9)
        + _s(fld.units, 25)
        + _s(fld.desc, 46)
        + struct.pack(">f3i", fld.level, nx, ny, 0),
    )
    _record(
        f,
        _s("SWCORNER", 8)
        + struct.pack(">5f", fld.startlat, fld.startlon, fld.dlat, fld.dlon, EARTH_RADIUS_KM),
    )
    _record(f, struct.pack(">i", 1 if fld.earth_relative_wind else 0))
    # C-order (ny, nx) bytes == Fortran-order (nx, ny)
    _record(f, data.tobytes(order="C"))


def write_file(path: str | Path, fields: list[Field]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        for fld in fields:
            write_field(f, fld)
    return path


def file_name(prefix: str, dt: datetime) -> str:
    """Name metgrid expects: PREFIX:YYYY-MM-DD_HH."""
    return f"{prefix}:{dt.strftime('%Y-%m-%d_%H')}"


def _read_record(f: BinaryIO) -> bytes | None:
    head = f.read(4)
    if not head:
        return None
    (n,) = struct.unpack(">i", head)
    payload = f.read(n)
    (tail,) = struct.unpack(">i", f.read(4))
    if tail != n:
        raise ValueError("corrupt Fortran record markers")
    return payload


def iter_fields(path: str | Path) -> Iterator[Field]:
    with open(path, "rb") as f:
        while True:
            rec = _read_record(f)
            if rec is None:
                return
            (ifv,) = struct.unpack(">i", rec)
            if ifv != IFV:
                raise ValueError(f"unsupported intermediate format version {ifv}")
            rec = _read_record(f)
            date = rec[0:24].decode().strip()
            (xfcst,) = struct.unpack(">f", rec[24:28])
            map_source = rec[28:60].decode().strip()
            name = rec[60:69].decode().strip()
            units = rec[69:94].decode().strip()
            desc = rec[94:140].decode().strip()
            xlvl, nx, ny, iproj = struct.unpack(">f3i", rec[140:156])
            if iproj != 0:
                raise NotImplementedError(f"projection {iproj} not supported")
            rec = _read_record(f)
            startloc = rec[0:8].decode().strip()
            startlat, startlon, dlat, dlon, _ = struct.unpack(">5f", rec[8:28])
            if startloc != "SWCORNER":
                raise NotImplementedError(f"startloc {startloc} not supported")
            (wind_rel,) = struct.unpack(">i", _read_record(f))
            slab = np.frombuffer(_read_record(f), dtype=">f4").reshape(ny, nx)
            yield Field(
                name=name,
                units=units,
                desc=desc,
                level=xlvl,
                date=datetime.strptime(date, "%Y-%m-%d_%H:%M:%S"),
                startlat=startlat,
                startlon=startlon,
                dlat=dlat,
                dlon=dlon,
                data=slab.astype(np.float32),
                map_source=map_source,
                xfcst=xfcst,
                earth_relative_wind=bool(wind_rel),
            )


def read_file(path: str | Path) -> list[Field]:
    return list(iter_fields(path))
