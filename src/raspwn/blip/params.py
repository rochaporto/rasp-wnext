"""Soaring (BLIPMAP-style) parameters from a WRF column state.

Open re-implementation of the classic RASP/BLIPMAP fields. DrJack's own
Fortran routines (``ncl_jack_fortran``) only circulate as binaries, so
these follow the published parameter descriptions and the formulas that
are visible in the RASP NCL drivers (vhf, bsratio, hglider, stars, pfd).
Where the original formula is not public (hcrit's thermal profile) the
choice made is documented in the function.

Conventions: 3D arrays are (nz, ny, nx), bottom-up; heights in metres
MSL unless the name says ``agl``; SI units everywhere (conversion to
ft/kt/fpm is a display concern).
"""

from __future__ import annotations

import numpy as np

from .. import thermo

FPM = 0.00508  # 1 ft/min in m/s
CDBL = 0.003  # boundary-layer drag coefficient used by RASP for B/S ratio
CWBASE_CRITERIA = 1.0e-5  # cloud-water mixing ratio criterion [kg/kg]

# ----------------------------------------------------------------------------
# column helpers
# ----------------------------------------------------------------------------


def bl_average(field: np.ndarray, zfull_agl: np.ndarray, pblh: np.ndarray) -> np.ndarray:
    """Thickness-weighted mean of a mass-level field over [0, pblh] AGL.

    zfull_agl: (nz+1, ny, nx) heights of the layer interfaces above ground.
    """
    bot, top = zfull_agl[:-1], zfull_agl[1:]
    w = np.clip(np.minimum(top, pblh) - bot, 0.0, None)
    wsum = w.sum(axis=0)
    # BL thinner than the first layer: use the lowest level
    return np.where(wsum > 0, (field * w).sum(axis=0) / np.where(wsum > 0, wsum, 1), field[0])


def bl_max(field: np.ndarray, z_agl: np.ndarray, pblh: np.ndarray) -> np.ndarray:
    inside = z_agl <= np.maximum(pblh, z_agl[0])
    return np.where(inside, field, -np.inf).max(axis=0)


def value_at(field: np.ndarray, coord: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Linearly interpolate ``field`` to ``target`` along axis 0 of ``coord``.

    ``coord`` must increase with k (e.g. height); targets outside the
    column are clamped to the end values.
    """
    nz = coord.shape[0]
    k = np.clip((coord <= target).sum(axis=0) - 1, 0, nz - 2)
    c0 = np.take_along_axis(coord, k[None], 0)[0]
    c1 = np.take_along_axis(coord, k[None] + 1, 0)[0]
    f0 = np.take_along_axis(field, k[None], 0)[0]
    f1 = np.take_along_axis(field, k[None] + 1, 0)[0]
    a = np.clip((target - c0) / np.where(c1 != c0, c1 - c0, 1.0), 0.0, 1.0)
    return f0 + a * (f1 - f0)


def height_at_pressure(z: np.ndarray, p: np.ndarray, ptarget: np.ndarray) -> np.ndarray:
    """Height where the column pressure equals ``ptarget`` (log-p interpolation)."""
    return value_at(z, -np.log(p), -np.log(ptarget))


# ----------------------------------------------------------------------------
# thermals
# ----------------------------------------------------------------------------


def virtual_heat_flux(hfx: np.ndarray, lh: np.ndarray, t_sfc: np.ndarray) -> np.ndarray:
    """Surface virtual (buoyancy) heat flux [W/m2], as in RASP.

    0.61 * cp / Lv * T * LH, with LH limited to >= 0 like RASP does.
    """
    return hfx + 0.000245268 * t_sfc * np.maximum(lh, 0.0)


def wstar(hfx, lh, t_sfc, psfc, pblh) -> np.ndarray:
    """Deardorff convective velocity scale W* [m/s]."""
    vhf = virtual_heat_flux(hfx, lh, t_sfc)
    rho = psfc / (thermo.RD * t_sfc)
    buoy = thermo.G / t_sfc * np.maximum(vhf, 0.0) / (rho * thermo.CP)
    return np.cbrt(buoy * np.maximum(pblh, 0.0))


# Thermal velocity profile: shape of Allen (2006, NASA/TM-2006-214318)
# s(x) = x^(1/3) (1 - 1.1 x), x = z/zi, rescaled so its peak equals W*.
# DrJack's exact hcrit profile is not published; calibrate if needed.
_X = np.linspace(0.0, 1.0 / 1.1, 2001)
_S = np.cbrt(_X) * (1.0 - 1.1 * _X)
_S /= _S.max()
_IPEAK = int(_S.argmax())
_X_DESC, _S_DESC = _X[_IPEAK:][::-1], _S[_IPEAK:][::-1]  # s ascending for np.interp


def thermal_profile(x: np.ndarray) -> np.ndarray:
    """Normalised updraft (w / W*) at relative height x = z / zi."""
    return np.interp(x, _X, _S, right=0.0)


def hcrit(ws: np.ndarray, ter: np.ndarray, pblh: np.ndarray, wcrit: float = 225 * FPM) -> np.ndarray:
    """Height [m MSL] where the thermal updraft drops below ``wcrit``.

    Default criterion 225 ft/min as in BLIPMAP's ``hwcrit``. Returns the
    terrain height where thermals never reach the criterion.
    """
    ratio = wcrit / np.maximum(ws, 1e-6)
    x = np.interp(ratio, _S_DESC, _X_DESC)
    return np.where(ratio <= 1.0, ter + x * pblh, ter)


# ----------------------------------------------------------------------------
# clouds
# ----------------------------------------------------------------------------


def sfc_lcl(t2, q2, psfc, z, p) -> np.ndarray:
    """Cumulus cloudbase [m MSL] for a surface parcel (RASP ``zsfclcl``)."""
    td = thermo.dewpoint_from_mixing_ratio(q2, psfc)
    plcl = thermo.lcl_pressure(t2, td, psfc)
    return height_at_pressure(z, p, plcl)


def bl_lcl(qv: np.ndarray, zfull_agl, z, t, p, pblh) -> np.ndarray:
    """Overdevelopment cloudbase [m MSL] (RASP ``zblcl``).

    Lowest height where the BL-averaged vapour mixing ratio saturates at
    the ambient temperature/pressure. Never-saturated columns get +inf.
    """
    qbl = bl_average(qv, zfull_agl, pblh)
    deficit = thermo.sat_mixing_ratio(t, p) - qbl  # >0 unsaturated
    sat = deficit <= 0
    k = np.argmax(sat, axis=0)
    found = sat.any(axis=0)
    km = np.maximum(k - 1, 0)
    d0 = np.take_along_axis(deficit, km[None], 0)[0]
    d1 = np.take_along_axis(deficit, k[None], 0)[0]
    z0 = np.take_along_axis(z, km[None], 0)[0]
    z1 = np.take_along_axis(z, k[None], 0)[0]
    a = np.where(k > 0, d0 / np.where(d0 != d1, d0 - d1, 1.0), 0.0)
    return np.where(found, z0 + np.clip(a, 0, 1) * (z1 - z0), np.inf)


def cloud_layers(cldfra: np.ndarray, p: np.ndarray) -> dict[str, np.ndarray]:
    """Low/mid/high cloud cover [0-1], maximum overlap within each band."""
    bands = {"cfracl": p >= 80000.0, "cfracm": (p < 80000.0) & (p >= 45000.0), "cfrach": p < 45000.0}
    return {k: np.where(m, cldfra, 0.0).max(axis=0) for k, m in bands.items()}


# ----------------------------------------------------------------------------
# composite indices
# ----------------------------------------------------------------------------


def bsratio(ws: np.ndarray, blwind: np.ndarray) -> np.ndarray:
    """Buoyancy/shear ratio: W* / sqrt(Cd U_bl^2) (RASP, metric units)."""
    return ws / np.sqrt(CDBL * np.maximum(blwind, 0.1) ** 2)


def _smooth(a: np.ndarray, weights) -> np.ndarray:
    w = np.asarray(weights, float)
    w /= w.sum()
    r = len(w) // 2
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        ap = np.pad(a, pad, mode="edge")
        a = sum(w[i] * np.take(ap, np.arange(i, i + a.shape[axis]), axis=axis) for i in range(len(w)))
    return a


def stars(hglider_agl, cupot, blwind, bs) -> np.ndarray:
    """RASP 0-5 "stars" rating for sailplanes (heuristic, UK-tuned).

    hglider_agl: usable height AGL [m]; cupot: cumulus potential [m];
    blwind: BL mean wind [m/s]; bs: buoyancy/shear ratio.
    """
    h = hglider_agl * 3.28084
    cu = cupot * 3.28084
    kt = blwind * 1.9438445
    s = np.select(
        [h < 1500, h <= 2500, h <= 3000, h <= 3500, h <= 4000],
        [0.0, 1000.0, 2000.0, 3000.0, 4000.0],
        5000.0,
    )
    s = np.where((cu > 500) & (cu < 5000), s + 500, s)
    s = np.where(cu < 0, s - 500, s)
    maxwind = 15.0
    kw = np.maximum(kt, maxwind)
    s = np.maximum(s - 250 * (kw - maxwind) ** 2 / kw, 0)
    s = np.maximum(s + (np.minimum(bs, 6.0) - 3.0) * 300, 0)
    return _smooth(s, [1, 6, 15, 20, 15, 6, 1]) / 1000.0


# Default polar (speed km/h -> sink m/s, negative down), from RASP's pfd.
POLAR = (-0.000302016415111608, 0.0729181700942301, -5.21488508245734)


def pfd_speed(ws, dglider, cupot, bs, sink=1.0, polar=POLAR) -> np.ndarray:
    """Achievable cross-country speed [km/h] for one hour (RASP ``pfd``).

    Climb = W* - sink, discounted by logistic "risk" factors for low
    working height, blue/overdeveloping skies and shear; then the
    MacCready speed-to-fly on ``polar``. Summing over the hours of a day
    gives the potential flight distance in km.
    """
    a, b, c = polar

    def risk(x, amp, fac, offx, offy):
        return np.maximum(amp / (1.0 + np.exp(-(x - offx) / fac)) + offy, 0.0)

    w = np.maximum(ws - sink, 0.0)
    w = w * risk(dglider, 2.0, 200.0, 300.0, -1.0) * risk(cupot, 0.3, 50.0, -100.0, 0.7)
    w = w * risk(bs, 2.0, 1.5, 1.0, -1.0)
    v = np.sqrt((c - w) / a)
    sink_v = a * v * v + b * v + c
    return np.where(w > 0, v * w / (w - sink_v), 0.0)
