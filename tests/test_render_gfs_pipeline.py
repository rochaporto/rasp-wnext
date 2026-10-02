from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from raspwn import config, pipeline
from raspwn.blip import render
from raspwn.blip.wrfout import Grid
from raspwn.wps import gfs

ROOT = Path(__file__).resolve().parents[1]


def lambert_grid(ny=40, nx=50, t1=38.0, t2=41.5, stand=-8.0):
    """Lat/lon of a regular Lambert grid, via the inverse projection."""
    t1r, t2r = np.radians(t1), np.radians(t2)
    cone = (np.log(np.cos(t1r)) - np.log(np.cos(t2r))) / (
        np.log(np.tan(np.pi / 4 - t1r / 2)) - np.log(np.tan(np.pi / 4 - t2r / 2)))
    rm0 = np.tan(np.pi / 4 - np.radians(39.8) / 2) ** cone
    d = 0.0015
    jj, ii = np.mgrid[0:ny, 0:nx]
    x = (ii - nx / 2) * d
    y = -rm0 + (jj - ny / 2) * d
    rm = np.hypot(x, y)
    lon = stand + np.degrees(np.arctan2(x, -y) / cone)
    lat = 90 - 2 * np.degrees(np.arctan(rm ** (1 / cone)))
    attrs = dict(MAP_PROJ=1, TRUELAT1=t1, TRUELAT2=t2, STAND_LON=stand)
    return Grid(lat, lon, attrs), ii


def test_resampler_lambert_is_exact():
    grid, ii = lambert_grid()
    rs = render.Resampler(grid)
    out = rs(grid.lon)
    inside = np.isfinite(out)
    assert inside.mean() > 0.6
    # resampled longitude equals the target column longitude
    target_lon = np.linspace(rs.west, rs.east, out.shape[1])[None, :].repeat(out.shape[0], 0)
    np.testing.assert_allclose(out[inside], target_lon[inside], atol=1e-3)
    rgba = render.colourise(rs(ii.astype(float)), "wstar")
    assert rgba.shape == out.shape + (4,) and (rgba[..., 3][~inside] == 0).all()


def test_gfs_idx_and_crop():
    idx = "1:0:d=2026100200:LAND:surface:anl:\n2:100:d=2026100200:ICEC:surface:anl:\n3:250:x:HGT:surface:anl:\n"
    e = gfs.parse_idx(idx)
    assert e[0][:2] == (0, 99) and e[2][:2] == (250, None)

    vals = np.arange(10 * 36, dtype=float).reshape(10, 36)  # 10 deg grid, lon 0..350
    grid = dict(startlat=-45.0, startlon=0.0, dlat=10.0, dlon=10.0)
    out, g = gfs._crop(vals, grid, (-25.0, -10.0, 15.0, 20.0))
    assert g["startlon"] == -30.0 and out.shape[1] == 6
    np.testing.assert_array_equal(out[0, :3], vals[3, 33:36])


def test_gfs_decode_message():
    eccodes = pytest.importorskip("eccodes")
    h = eccodes.codes_grib_new_from_samples("regular_ll_sfc_grib2")
    ni, nj = eccodes.codes_get(h, "Ni"), eccodes.codes_get(h, "Nj")
    vals = np.arange(ni * nj, dtype=float)
    eccodes.codes_set_values(h, vals)
    msg = eccodes.codes_get_message(h)
    eccodes.codes_release(h)
    data, grid = gfs._decode(msg)
    assert data.shape == (nj, ni)
    # stored north->south; decoded south->north
    np.testing.assert_allclose(data[-1], vals[:ni])
    assert grid["dlat"] > 0


def test_namelists_render():
    cfg = config.load(ROOT / "regions" / "example" / "region.yaml")
    run = pipeline.Run(cfg, datetime(2026, 10, 2, 0), day=1)
    assert run.start == datetime(2026, 10, 3, 6) and run.end == datetime(2026, 10, 3, 21)
    wps = pipeline.render(cfg, "namelist.wps.j2", run.ctx())
    assert "constants_name  = 'GFSLSM:2026-10-03_06'" in wps
    assert "start_date       = '2026-10-03_06:00:00','2026-10-03_06:00:00'," in wps
    assert "parent_id         = 1,1," in wps
    nml = pipeline.render(cfg, "namelist.input.j2", run.ctx())
    assert "dx                     = 9000,3000.0," in nml
    assert "num_metgrid_levels     = 14," in nml
    assert "cu_physics         = 3,0," in nml
    assert "interval_seconds   = 3600" in nml


def test_namelists_end_with_newline():
    cfg = config.load(ROOT / "regions" / "example" / "region.yaml")
    run = pipeline.Run(cfg, datetime(2026, 10, 2, 0))
    for name in ("namelist.wps.j2", "namelist.input.j2"):
        assert pipeline.render(cfg, name, run.ctx()).endswith("/\n")
