from datetime import datetime, timedelta

import numpy as np
import pytest

from conftest import INIT, std_atmosphere
from raspwn import thermo
from raspwn.wnext.source import LEVELS_HPA, WeatherNextConfig, normalise
from raspwn.wps import convert
from raspwn.wps.intermediate import read_file

BBOX = (-12.0, 36.0, 4.0, 44.0)  # crosses the Greenwich meridian
START = INIT + timedelta(hours=6)
END = INIT + timedelta(hours=21)


def test_normalise_layout(raw_wn3):
    up, sf = normalise(raw_wn3, WeatherNextConfig(member=1), INIT, START, END, BBOX)

    assert dict(up.sizes) == {"time": 16, "level": 13, "lat": 5, "lon": 9}
    assert up.time.values[0] == np.datetime64(START)
    assert up.time.values[-1] == np.datetime64(END)
    assert np.all(np.diff(up.lon.values) > 0) and up.lon.values[0] == -12.0
    assert np.all(np.diff(up.lat.values) > 0)
    assert list(up.level.values[-3:]) == [850, 925, 1000]
    # value = ISA + 0.01 * valid hour + member index
    t_isa, _ = std_atmosphere([850])
    hour = 9
    got = float(up.t.sel(level=850).isel(time=hour - 6)[0, 0])
    assert got == pytest.approx(t_isa[0] + 0.01 * hour + 1, abs=1e-3)
    assert float(sf.t2m.isel(time=0)[0, 0]) == pytest.approx(296.0)
    assert sf.lon.values[0] == -12.0 and sf.sizes["lon"] == 17


def test_ensemble_mean(raw_wn3):
    up, _ = normalise(raw_wn3, WeatherNextConfig(member="mean"), INIT, START, END, BBOX)
    t_isa, _ = std_atmosphere([850])
    assert float(up.t.sel(level=850).isel(time=0)[0, 0]) == pytest.approx(t_isa[0] + 0.06 + 0.5, abs=1e-3)


def test_surface_pressure_matches_isa():
    levels = np.array(LEVELS_HPA) * 100.0
    _, z = std_atmosphere(levels / 100.0)
    ght = np.broadcast_to(z[:, None, None], (levels.size, 1, 3))
    pmsl = np.full((1, 3), 101325.0)
    terrain = np.array([[0.0, 500.0, 1500.0]])
    ps = convert.surface_pressure(ght, levels, pmsl, terrain)
    isa = 101325.0 * (1 - 2.25577e-5 * terrain) ** 5.25588
    np.testing.assert_allclose(ps, isa, rtol=2e-3)


def test_write_intermediate(raw_wn3, tmp_path):
    up, sf = normalise(raw_wn3, WeatherNextConfig(), INIT, START, START + timedelta(hours=1), BBOX)
    terrain = (np.zeros((40, 40)), dict(startlat=30.0, startlon=-20.0, dlat=0.5, dlon=0.5))
    paths = convert.write_intermediate(up, sf, tmp_path, terrain=terrain)
    assert [p.name for p in paths] == ["WNEXT:2026-10-02_06", "WNEXT:2026-10-02_07"]

    fields = read_file(paths[0])
    names = {(f.name, f.level) for f in fields}
    for var in ("TT", "UU", "VV", "RH", "GHT"):
        assert (var, 85000.0) in names
    for var in ("TT", "UU", "VV", "RH", "SST", "PSFC"):
        assert (var, 200100.0) in names
    assert ("PMSL", 201300.0) in names
    assert len(fields) == 13 * 5 + 7

    rh850 = next(f for f in fields if f.name == "RH" and f.level == 85000.0)
    t850 = next(f for f in fields if f.name == "TT" and f.level == 85000.0)
    expect = thermo.rh_from_q(0.005 * 0.85, t850.data, 85000.0)
    np.testing.assert_allclose(rh850.data, expect, rtol=1e-5)
    psfc = next(f for f in fields if f.name == "PSFC")
    np.testing.assert_allclose(psfc.data, 101325.0, rtol=1e-3)  # sea-level terrain
    assert psfc.dlat == 2.0 and next(f for f in fields if f.name == "PMSL").dlat == 1.0


def test_missing_time_coverage_detected(raw_wn3):
    late = datetime(2026, 10, 3, 6)
    with pytest.raises(ValueError, match="no common valid times"):
        normalise(raw_wn3, WeatherNextConfig(), INIT, late, late + timedelta(hours=3), BBOX)
