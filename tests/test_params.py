import numpy as np
import pytest

from raspwn import thermo
from raspwn.blip import params as bp
from raspwn.blip import wrfout


def column(nz=40, ny=2, nx=3, dz=100.0, ter=200.0):
    """Dry-adiabatic mixed layer to 1500 m AGL, stable above."""
    zfull = ter + np.arange(nz + 1)[:, None, None] * dz * np.ones((1, ny, nx))
    z = 0.5 * (zfull[1:] + zfull[:-1])
    agl = z - ter
    theta = np.where(agl < 1500, 300.0, 300.0 + 0.005 * (agl - 1500))
    # hydrostatic pressure from theta
    p = np.empty_like(z)
    pi = (100000.0 / 1e5) ** thermo.KAPPA
    for k in range(nz):
        dzk = z[k] - (ter if k == 0 else z[k - 1])
        pi = pi - thermo.G * dzk / (thermo.CP * theta[k])
        p[k] = 1e5 * pi ** (1 / thermo.KAPPA)
    t = theta * (p / 1e5) ** thermo.KAPPA
    return zfull, z, p, t


def test_bl_average_and_value_at():
    zfull, z, _, _ = column()
    ter = 200.0
    f = z - ter  # field = height AGL
    pblh = np.full(z.shape[1:], 1000.0)
    np.testing.assert_allclose(bp.bl_average(f, zfull - ter, pblh), 500.0)
    np.testing.assert_allclose(bp.value_at(f, z - ter, pblh), 1000.0)


def test_wstar_typical_value():
    # 300 W/m2 sensible, dry, 1500 m BL -> about 2.2 m/s
    ws = bp.wstar(np.array(300.0), np.array(0.0), np.array(300.0), np.array(1e5), np.array(1500.0))
    rho = 1e5 / (thermo.RD * 300.0)
    expect = (9.80665 / 300.0 * 300.0 / (rho * thermo.CP) * 1500.0) ** (1 / 3)
    assert ws == pytest.approx(expect) and 2.0 < ws < 2.5


def test_hcrit():
    ter = np.zeros(3)
    pblh = np.full(3, 2000.0)
    h = bp.hcrit(np.array([0.5, 2.0, 4.0]), ter, pblh)
    assert h[0] == 0.0  # weak thermals never reach 225 fpm
    assert 0 < h[1] < h[2] < 2000.0 / 1.1
    # profile peak equals W*
    assert bp.thermal_profile(np.linspace(0, 1, 1001)).max() == pytest.approx(1.0)


def test_cloudbase():
    zfull, z, p, t = column()
    ter = 200.0
    ny, nx = z.shape[1:]
    psfc = np.full((ny, nx), 1e5)
    t2 = np.full((ny, nx), 300.0)
    td = np.full((ny, nx), 290.0)
    q2 = thermo.mixing_ratio_from_e(thermo.sat_vapour_pressure(td), psfc)
    lcl = bp.sfc_lcl(t2, q2, psfc, z, p)
    # rule of thumb: ~122-125 m per K of dew point depression
    np.testing.assert_allclose(lcl - ter, 1240.0, rtol=0.05)

    qv = np.broadcast_to(q2, z.shape)
    pblh = np.full((ny, nx), 1500.0)
    zbl = bp.bl_lcl(qv, zfull - ter, z, t, p, pblh)
    np.testing.assert_allclose(zbl, lcl, atol=60.0)  # well-mixed: same parcel


def test_bl_lcl_never_saturated():
    zfull, z, p, t = column()
    qv = np.full(z.shape, 1e-5)
    out = bp.bl_lcl(qv, zfull - 200.0, z, t, p, np.full(z.shape[1:], 1500.0))
    assert np.all(np.isinf(out))


def test_bsratio_stars_pfd():
    assert bp.bsratio(np.array(2.0), np.array(5.0)) == pytest.approx(2.0 / (np.sqrt(0.003) * 5.0))
    good = bp.stars(np.full((9, 9), 1500.0), np.full((9, 9), 800.0), np.full((9, 9), 3.0),
                    np.full((9, 9), 8.0))
    bad = bp.stars(np.full((9, 9), 300.0), np.full((9, 9), -800.0), np.full((9, 9), 15.0),
                   np.full((9, 9), 1.0))
    assert good.min() > 4.0 and bad.max() == 0.0
    v = bp.pfd_speed(np.array([0.5, 2.5, 3.5]), np.array([1500.0] * 3), np.array([800.0] * 3),
                     np.array([8.0] * 3))
    assert v[0] == 0.0 and 40 < v[1] < v[2] < 150


def test_compute_from_state():
    zfull, z, p, t = column()
    ny, nx = z.shape[1:]
    shape2 = (ny, nx)
    theta = t * (1e5 / p) ** thermo.KAPPA
    s = dict(
        ter=np.full(shape2, 200.0), pblh=np.full(shape2, 1500.0), hfx=np.full(shape2, 250.0),
        lh=np.full(shape2, 100.0), t2=np.full(shape2, 300.0), q2=np.full(shape2, 0.008),
        psfc=np.full(shape2, 1e5), u10=np.full(shape2, 3.0), v10=np.zeros(shape2),
        swdown=np.full(shape2, 800.0), rain=np.zeros(shape2), zfull=zfull, z=z, p=p,
        theta=theta, t=t, qv=np.full(z.shape, 0.008), u=np.full(z.shape, 5.0),
        v=np.zeros(z.shape), cldfra=np.zeros(z.shape),
    )
    out = wrfout.compute(s)
    for name in wrfout.PARAMETERS:
        assert name in out, name
        assert np.all(np.isfinite(out[name])) or name == "zblcl", name
    assert np.all(out["hglider"] <= out["hbl"] + 1e-6)
    assert np.all(out["dglider"] >= 0)
    np.testing.assert_allclose(out["blwind"], 5.0)
