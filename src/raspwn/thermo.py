"""Small set of moist thermodynamics helpers (SI units, numpy-vectorised)."""

from __future__ import annotations

import numpy as np

G = 9.80665
RD = 287.04
CP = 1004.5
LV = 2.501e6
EPS = 0.622
KAPPA = RD / CP


def sat_vapour_pressure(t_k):
    """Saturation vapour pressure over water [Pa] (Bolton 1980)."""
    tc = np.asarray(t_k) - 273.15
    return 611.2 * np.exp(17.67 * tc / (tc + 243.5))


def vapour_pressure_from_q(q, p):
    q = np.asarray(q)
    return q * p / (EPS + (1.0 - EPS) * q)


def rh_from_q(q, t_k, p):
    """Relative humidity [%] from specific humidity, temperature, pressure."""
    rh = 100.0 * vapour_pressure_from_q(q, p) / sat_vapour_pressure(t_k)
    return np.clip(rh, 0.0, 100.0)


def rh_from_dewpoint(t_k, td_k):
    rh = 100.0 * sat_vapour_pressure(td_k) / sat_vapour_pressure(t_k)
    return np.clip(rh, 0.0, 100.0)


def mixing_ratio_from_e(e, p):
    return EPS * e / np.maximum(p - e, 1.0)


def sat_mixing_ratio(t_k, p):
    return mixing_ratio_from_e(sat_vapour_pressure(t_k), p)


def dewpoint_from_mixing_ratio(w, p):
    """Dew point [K] from water vapour mixing ratio [kg/kg] and pressure [Pa]."""
    e = np.maximum(np.asarray(w) * p / (EPS + np.asarray(w)), 1e-3)
    x = np.log(e / 611.2)
    return 273.15 + 243.5 * x / (17.67 - x)


def lcl_temperature(t_k, td_k):
    """Temperature at the lifting condensation level [K] (Bolton 1980, eq. 15)."""
    td_k = np.minimum(td_k, t_k)
    return 1.0 / (1.0 / (td_k - 56.0) + np.log(t_k / td_k) / 800.0) + 56.0


def lcl_pressure(t_k, td_k, p):
    """Pressure at the LCL [Pa] for a dry-adiabatic ascent."""
    return p * (lcl_temperature(t_k, td_k) / t_k) ** (1.0 / KAPPA)
