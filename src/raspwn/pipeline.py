"""End-to-end forecast run: WeatherNext -> WPS -> WRF -> soaring maps.

Run directory layout (``wrf.workdir``):

    <region>/geo/                     geogrid output, reused across runs
    <region>/<YYYYMMDDHH>_d<day>_m<member>/
        GFSLSM:*  WNEXT:*             intermediate files (Python-written)
        met_em.*                      metgrid output
        wrfout_*                      WRF output
        *.log
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

import jinja2
import numpy as np

from . import config as cfgmod
from .wnext import source
from .wps import convert, gfs

log = logging.getLogger(__name__)

PHYSICS_DEFAULTS = dict(
    mp_physics=8,  # Thompson
    ra_lw_physics=4,  # RRTMG
    ra_sw_physics=4,
    radt=10,
    sf_sfclay_physics=1,  # revised MM5
    sf_surface_physics=2,  # Noah
    bl_pbl_physics=1,  # YSU
    cu_physics=3,  # Grell-Freitas, only on grids > 5 km
)


# Lookup tables Thompson microphysics computes on first use (~5 min)
THOMPSON_TABLES = ("qr_acr_qg_V4.dat", "qr_acr_qsV2.dat", "freezeH2O.dat")


class StepFailed(RuntimeError):
    pass


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def wps_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d_%H:%M:%S")


def domains_ctx(region: cfgmod.Region) -> list[dict]:
    out = []
    for n, d in enumerate(region.domains):
        dd = asdict(d)
        dd["parent_id"] = max(n, 1)
        dd["dx"] = d.dx if n == 0 else out[-1]["dx"] / d.ratio
        out.append(dd)
    return out


def render(cfg: cfgmod.Config, name: str, ctx: dict) -> str:
    dirs = [cfg.wrf.templates] if cfg.wrf.templates else []
    dirs.append(str(Path(__file__).parent / "templates"))
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(dirs),
        undefined=jinja2.StrictUndefined,
        keep_trailing_newline=True,  # Fortran namelist reads hit EOF without it
    )
    return env.get_template(name).render(**ctx)


TOOL_LOGS = ("rsl.error.0000", "metgrid.log", "geogrid.log")


def run_exe(cmd: list[str], cwd: Path, log_name: str, ok_marker: str | None = None) -> None:
    logfile = cwd / log_name
    for stale in TOOL_LOGS:  # an old success marker must not hide a new failure
        (cwd / stale).unlink(missing_ok=True)
    log.info("running %s in %s", " ".join(cmd), cwd)
    with open(logfile, "w") as fh:
        res = subprocess.run(cmd, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT)
    text = logfile.read_text(errors="replace")
    # WRF tools also write rsl.* / *.log files; check those for the marker too
    for extra in TOOL_LOGS:
        if (cwd / extra).exists():
            text += (cwd / extra).read_text(errors="replace")
    if res.returncode != 0 or (ok_marker and ok_marker not in text):
        raise StepFailed(f"{cmd[0]} failed (exit {res.returncode}); see {logfile}")


def link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        dst.unlink()
    dst.symlink_to(src)


def bbox_from_geo(geo_file: Path) -> tuple[float, float, float, float]:
    from netCDF4 import Dataset

    with Dataset(geo_file) as nc:
        lat = nc.variables["XLAT_M"][0]
        lon = nc.variables["XLONG_M"][0]
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


# ----------------------------------------------------------------------------
# run
# ----------------------------------------------------------------------------


class Run:
    def __init__(self, cfg: cfgmod.Config, init: datetime, day: int = 0):
        self.cfg = cfg
        self.init = init
        self.day = day
        f = cfg.forecast
        self.start = init + timedelta(hours=f.start_offset_hours + 24 * day)
        self.end = self.start + timedelta(hours=f.length_hours)
        base = Path(cfg.wrf.workdir) / cfg.region.name
        self.geo_dir = base / "geo"
        member = cfg.weathernext.member
        self.dir = base / f"{init:%Y%m%d%H}_d{day}_m{member}"
        self.interval = 3600

    # -- context for the namelist templates
    def ctx(self) -> dict:
        domains = domains_ctx(self.cfg.region)
        return dict(
            region=self.cfg.region,
            domains=domains,
            start=wps_date(self.start),
            end=wps_date(self.end),
            start_dt=self.start,
            end_dt=self.end,
            interval=self.interval,
            geog_data_path=self.cfg.wrf.geog_data_path,
            constants=f"{gfs.PREFIX}:{self.start:%Y-%m-%d_%H}",
            time_step=int(5 * domains[0]["dx"] / 1000),
            e_vert=self.cfg.wrf.e_vert,
            num_metgrid_levels=len(self.cfg.weathernext.levels) + 1,
            physics={**PHYSICS_DEFAULTS, **self.cfg.wrf.physics},
        )

    def _write_wps_namelist(self, where: Path) -> None:
        where.mkdir(parents=True, exist_ok=True)
        (where / "namelist.wps").write_text(render(self.cfg, "namelist.wps.j2", self.ctx()))

    # -- steps
    def geogrid(self, force: bool = False) -> None:
        if not force and (self.geo_dir / "geo_em.d01.nc").exists():
            log.info("geogrid: reusing %s", self.geo_dir)
            return
        wps = Path(self.cfg.wrf.wps_dir)
        self._write_wps_namelist(self.geo_dir)
        (self.geo_dir / "geogrid").mkdir(exist_ok=True)
        link(wps / "geogrid" / "GEOGRID.TBL.ARW", self.geo_dir / "geogrid" / "GEOGRID.TBL")
        run_exe([str(wps / "geogrid.exe")], self.geo_dir, "geogrid.out",
                "Successful completion of geogrid")

    def fetch(self) -> None:
        """Write GFSLSM + WNEXT intermediate files for the run window."""
        self.dir.mkdir(parents=True, exist_ok=True)
        bbox = bbox_from_geo(self.geo_dir / "geo_em.d01.nc")
        log.info("domain bbox %s", bbox)

        lsm = gfs.fetch(self.init, self.start, bbox)
        gfs.write(lsm, self.dir)

        upper, surface = source.load(self.cfg.weathernext, self.init, self.start, self.end, bbox)
        times = upper.time.values
        steps = np.unique(np.diff(times).astype("timedelta64[s]").astype(int))
        if len(steps) != 1:
            raise StepFailed(f"WeatherNext valid times are not evenly spaced: {steps}")
        self.interval = int(steps[0])
        if times[0] > np.datetime64(self.start) or times[-1] < np.datetime64(self.end):
            raise StepFailed(f"WeatherNext covers {times[0]}..{times[-1]}, need {self.start}..{self.end}")
        convert.write_intermediate(upper, surface, self.dir, terrain=gfs.terrain(lsm))
        (self.dir / "interval").write_text(str(self.interval))

    def metgrid(self) -> None:
        wps = Path(self.cfg.wrf.wps_dir)
        if (self.dir / "interval").exists():
            self.interval = int((self.dir / "interval").read_text())
        self._write_wps_namelist(self.dir)
        (self.dir / "metgrid").mkdir(exist_ok=True)
        link(wps / "metgrid" / "METGRID.TBL.ARW", self.dir / "metgrid" / "METGRID.TBL")
        for geo in self.geo_dir.glob("geo_em.d*.nc"):
            link(geo, self.dir / geo.name)
        run_exe([str(wps / "metgrid.exe")], self.dir, "metgrid.out",
                "Successful completion of metgrid")

    def wrf(self) -> None:
        run_dir = Path(self.cfg.wrf.wrf_run_dir)
        for item in run_dir.iterdir():
            if not item.name.startswith("namelist") and not item.name.endswith(".exe"):
                link(item, self.dir / item.name)
        if (self.dir / "interval").exists():
            self.interval = int((self.dir / "interval").read_text())
        (self.dir / "namelist.input").write_text(render(self.cfg, "namelist.input.j2", self.ctx()))
        cache = Path(self.cfg.wrf.workdir) / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        for table in cache.glob("*.dat"):
            link(table, self.dir / table.name)

        mpirun = self.cfg.wrf.mpirun
        # real.exe single-process: with several MPI ranks (WRF 4.6.1, gfortran
        # 13, MPICH 4.2) the first wrfbdy time record comes out corrupted.
        run_exe([mpirun, "-np", "1", str(run_dir / "real.exe")], self.dir, "real.out",
                "SUCCESS COMPLETE REAL_EM")
        try:
            run_exe([mpirun, "-np", str(self.cfg.wrf.mpi_procs), str(run_dir / "wrf.exe")],
                    self.dir, "wrf.out", "SUCCESS COMPLETE WRF")
        finally:
            for name in THOMPSON_TABLES:
                path = self.dir / name
                if path.is_file() and not path.is_symlink():
                    shutil.copy(path, cache / name)

    def post(self) -> Path:
        from . import post

        return post.process(self.cfg, self.dir, self.init, self.day)

    def all(self) -> Path:
        self.geogrid()
        self.fetch()
        self.metgrid()
        self.wrf()
        return self.post()


def resolve_init(cfg: cfgmod.Config, init: str | None) -> datetime:
    value = init or cfg.forecast.init
    if value == "latest":
        return source.latest_init(cfg.weathernext)
    dt = datetime.fromisoformat(value)
    if dt.hour % 6:
        raise ValueError("WeatherNext pressure levels exist only for 00/06/12/18Z inits")
    return dt.replace(tzinfo=None)

