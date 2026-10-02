"""YAML run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml

from .blip.wrfout import PARAMETERS
from .wnext.source import WeatherNextConfig


@dataclass
class Domain:
    e_we: int
    e_sn: int
    dx: float = 0.0  # only for the outermost domain; nests derive it
    ratio: int = 1
    i_start: int = 1
    j_start: int = 1
    geog_data_res: str = "default"


@dataclass
class Region:
    name: str
    ref_lat: float
    ref_lon: float
    truelat1: float
    truelat2: float
    stand_lon: float
    domains: list[Domain]
    map_proj: str = "lambert"


@dataclass
class Forecast:
    init: str = "latest"  # or ISO time, e.g. 2026-10-02T00
    start_offset_hours: int = 6  # WRF start = init + offset (+ 24 * day)
    length_hours: int = 15
    days: list[int] = field(default_factory=lambda: [0])


@dataclass
class Wrf:
    wps_dir: str = "/opt/WPS"
    wrf_run_dir: str = "/opt/WRF/run"
    geog_data_path: str = "/data/geog"
    workdir: str = "/data/runs"
    mpi_procs: int = 4
    mpirun: str = "mpirun"
    e_vert: int = 55
    physics: dict = field(default_factory=dict)  # extra &physics overrides
    templates: str | None = None  # directory with custom *.j2 namelists


@dataclass
class Output:
    dir: str = "/data/out"
    params: list[str] = field(default_factory=lambda: list(PARAMETERS) + ["pfd_tot"])
    domain: int | None = None  # which WRF domain to render (default: innermost)


@dataclass
class Config:
    region: Region
    forecast: Forecast
    weathernext: WeatherNextConfig
    wrf: Wrf
    output: Output
    path: Path | None = None


def _build(cls, data: dict | None):
    data = dict(data or {})
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"{cls.__name__}: unknown keys {sorted(unknown)}")
    return cls(**data)


def load(path: str | Path) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    reg = dict(raw["region"])
    reg["domains"] = [_build(Domain, d) for d in reg["domains"]]
    if not reg["domains"][0].dx:
        raise ValueError("region.domains[0].dx is required")
    return Config(
        region=_build(Region, reg),
        forecast=_build(Forecast, raw.get("forecast")),
        weathernext=_build(WeatherNextConfig, raw.get("weathernext")),
        wrf=_build(Wrf, raw.get("wrf")),
        output=_build(Output, raw.get("output")),
        path=path,
    )
