import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from raspwn import config, post, thermo
from test_params import column
from test_render_gfs_pipeline import ROOT, lambert_grid


def write_wrfout(path: Path, valid: datetime, grid, rain: float) -> None:
    ny, nx = grid.lat.shape
    zfull, z, p, t = column(ny=ny, nx=nx)
    nz = z.shape[0]
    theta = t * (1e5 / p) ** thermo.KAPPA
    with Dataset(path, "w") as nc:
        for d, n in [("Time", None), ("DateStrLen", 19), ("bottom_top", nz),
                     ("bottom_top_stag", nz + 1), ("south_north", ny), ("west_east", nx),
                     ("south_north_stag", ny + 1), ("west_east_stag", nx + 1)]:
            nc.createDimension(d, n)
        nc.setncatts(grid.attrs)
        times = nc.createVariable("Times", "S1", ("Time", "DateStrLen"))
        times[0] = np.array(list(valid.strftime("%Y-%m-%d_%H:%M:%S")), dtype="S1")

        def var(name, dims, value):
            v = nc.createVariable(name, "f4", ("Time",) + dims)
            v[0] = np.broadcast_to(value, tuple(nc.dimensions[d].size for d in dims))

        d2 = ("south_north", "west_east")
        d3 = ("bottom_top",) + d2
        var("XLAT", d2, grid.lat)
        var("XLONG", d2, grid.lon)
        for name, val in dict(HGT=200.0, PBLH=1500.0, HFX=250.0, LH=80.0, T2=300.0, Q2=0.008,
                              PSFC=1e5, U10=3.0, V10=1.0, SWDOWN=800.0, RAINC=0.0,
                              RAINNC=rain, COSALPHA=1.0, SINALPHA=0.0).items():
            var(name, d2, val)
        var("PH", ("bottom_top_stag",) + d2, zfull * thermo.G)
        var("PHB", ("bottom_top_stag",) + d2, 0.0)
        var("P", d3, p)
        var("PB", d3, 0.0)
        var("T", d3, theta - 300.0)
        var("QVAPOR", d3, 0.008)
        var("CLDFRA", d3, 0.0)
        var("U", ("bottom_top", "south_north", "west_east_stag"), 6.0)
        var("V", ("bottom_top", "south_north_stag", "west_east"), 0.0)


def test_post_process(tmp_path):
    cfg = config.load(ROOT / "regions" / "example" / "region.yaml")
    cfg.output.dir = str(tmp_path / "out")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    grid, _ = lambert_grid(ny=20, nx=24)
    t0 = datetime(2026, 10, 2, 12)
    for h in range(3):
        valid = t0 + timedelta(hours=h)
        write_wrfout(run_dir / f"wrfout_d02_{valid:%Y-%m-%d_%H:%M:%S}", valid, grid, rain=float(h))
    # a coarser-domain file must be ignored
    write_wrfout(run_dir / f"wrfout_d01_{t0:%Y-%m-%d_%H:%M:%S}", t0, grid, 0.0)

    out = post.process(cfg, run_dir, datetime(2026, 10, 2, 0), 0)

    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["times"] == ["20261002T1200Z", "20261002T1300Z", "20261002T1400Z"]
    assert {"wstar", "hglider", "stars", "pfd_tot", "rain1"} <= set(manifest["params"])
    assert manifest["params"]["wstar"]["units"] == "m/s"
    for p in manifest["params"]:
        for t in manifest["times"]:
            assert (out / p / f"{t}.png").exists(), (p, t)
    root = Path(cfg.output.dir)
    assert json.loads((root / "runs.json").read_text()) == ["example/2026100200_d0_m0"]
    assert (root / "index.html").exists()
