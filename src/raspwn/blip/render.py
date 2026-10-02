"""Render parameter fields as web-map overlays.

WRF grids are Lambert conformal (or Mercator). Leaflet image overlays are
north-up rectangles in Web Mercator, so every field is resampled onto a
grid that is regular in Web Mercator x/y before being written as a PNG.
A ``manifest.json`` describes the times, parameters and colour scales for
``web/index.html``.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .wrfout import DAILY_PARAMETERS, PARAMETERS, Grid

# param -> (vmin, vmax, matplotlib colormap)
SCALES: dict[str, tuple[float, float, str]] = {
    "wstar": (0.0, 4.0, "turbo"),
    "bsratio": (0.0, 10.0, "viridis"),
    "hwcrit": (0.0, 4000.0, "turbo"),
    "dwcrit": (0.0, 3000.0, "turbo"),
    "hglider": (0.0, 4000.0, "turbo"),
    "dglider": (0.0, 3000.0, "turbo"),
    "hbl": (0.0, 4000.0, "turbo"),
    "dbl": (0.0, 3000.0, "turbo"),
    "zsfclcl": (0.0, 4000.0, "turbo"),
    "zsfclcldif": (-1000.0, 1500.0, "RdYlBu_r"),
    "zblcl": (0.0, 4000.0, "turbo"),
    "zblcldif": (-1000.0, 1500.0, "RdYlBu_r"),
    "stars": (0.0, 5.0, "turbo"),
    "pfd": (0.0, 120.0, "turbo"),
    "pfd_tot": (0.0, 800.0, "turbo"),
    "sfcsun": (0.0, 1000.0, "inferno"),
    "sfcshf": (-50.0, 500.0, "inferno"),
    "sfctemp": (-10.0, 40.0, "RdYlBu_r"),
    "sfcdewpt": (-20.0, 25.0, "BrBG"),
    "sfcwind": (0.0, 20.0, "viridis"),
    "blwind": (0.0, 20.0, "viridis"),
    "bltopwind": (0.0, 25.0, "viridis"),
    "blwindshear": (0.0, 15.0, "magma"),
    "blcloud": (0.0, 100.0, "Greys"),
    "cfracl": (0.0, 100.0, "Greys"),
    "cfracm": (0.0, 100.0, "Greys"),
    "cfrach": (0.0, 100.0, "Greys"),
    "rain1": (0.0, 10.0, "Blues"),
}

R_MERC = 6378137.0


def _merc_y(lat):
    return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def _merc_lat(y):
    return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)


def _projected_xy(lat, lon, attrs: dict) -> tuple[np.ndarray, np.ndarray]:
    """Map coordinates (up to an affine transform) in the WRF projection."""
    proj = int(attrs.get("MAP_PROJ", 1))
    lat_r = np.radians(lat)
    if proj == 1:  # Lambert conformal
        t1, t2 = np.radians(attrs["TRUELAT1"]), np.radians(attrs["TRUELAT2"])
        hemi = 1.0 if attrs["TRUELAT1"] >= 0 else -1.0
        if abs(t1 - t2) > 1e-3:
            cone = (np.log(np.cos(t1)) - np.log(np.cos(t2))) / (
                np.log(np.tan(np.pi / 4 - hemi * t1 / 2)) - np.log(np.tan(np.pi / 4 - hemi * t2 / 2))
            )
        else:
            cone = np.sin(abs(t1))
        dlon = ((lon - attrs["STAND_LON"] + 180.0) % 360.0) - 180.0
        rm = np.tan(np.pi / 4 - hemi * lat_r / 2) ** cone
        arg = cone * np.radians(dlon)
        return rm * np.sin(arg), -hemi * rm * np.cos(arg)
    if proj == 3:  # Mercator
        return np.radians(lon), _merc_y(lat)
    raise NotImplementedError(f"MAP_PROJ={proj} not supported for rendering")


class Resampler:
    """Bilinear resampling from the WRF grid to a Web-Mercator-regular grid."""

    def __init__(self, grid: Grid, width: int | None = None):
        ny, nx = grid.lat.shape
        x, y = _projected_xy(grid.lat, grid.lon, grid.attrs)
        jj, ii = np.mgrid[0:ny, 0:nx]
        # (i, j) as an affine function of projected (x, y)
        a = np.column_stack([x.ravel(), y.ravel(), np.ones(x.size)])
        coef, *_ = np.linalg.lstsq(a, np.column_stack([ii.ravel(), jj.ravel()]), rcond=None)

        self.west, self.east = float(grid.lon.min()), float(grid.lon.max())
        self.south, self.north = float(grid.lat.min()), float(grid.lat.max())
        width = width or int(nx * 1.5)
        ys = np.linspace(_merc_y(self.north), _merc_y(self.south), int(width * (
            (_merc_y(self.north) - _merc_y(self.south)) / np.radians(self.east - self.west))))
        lats = _merc_lat(ys)
        lons = np.linspace(self.west, self.east, width)
        tlon, tlat = np.meshgrid(lons, lats)
        tx, ty = _projected_xy(tlat, tlon, grid.attrs)
        ij = np.column_stack([tx.ravel(), ty.ravel(), np.ones(tx.size)]) @ coef
        self.fi = ij[:, 0].reshape(tlat.shape)
        self.fj = ij[:, 1].reshape(tlat.shape)
        self.inside = (self.fi >= 0) & (self.fi <= nx - 1) & (self.fj >= 0) & (self.fj <= ny - 1)

    @property
    def bounds(self) -> list[list[float]]:
        return [[self.south, self.west], [self.north, self.east]]

    def __call__(self, field: np.ndarray) -> np.ndarray:
        ny, nx = field.shape
        fi = np.clip(self.fi, 0, nx - 1.000001)
        fj = np.clip(self.fj, 0, ny - 1.000001)
        i0, j0 = fi.astype(int), fj.astype(int)
        wi, wj = fi - i0, fj - j0
        out = (
            field[j0, i0] * (1 - wi) * (1 - wj)
            + field[j0, i0 + 1] * wi * (1 - wj)
            + field[j0 + 1, i0] * (1 - wi) * wj
            + field[j0 + 1, i0 + 1] * wi * wj
        )
        return np.where(self.inside, out, np.nan)


def colourise(values: np.ndarray, param: str, alpha: int = 200) -> np.ndarray:
    import matplotlib

    vmin, vmax, cmap = SCALES.get(param, (np.nanmin(values), np.nanmax(values), "viridis"))
    norm = np.clip((values - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    rgba = (matplotlib.colormaps[cmap](np.nan_to_num(norm)) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isfinite(values), alpha, 0)
    return rgba


def legend(param: str, steps: int = 11) -> dict:
    import matplotlib

    vmin, vmax, cmap = SCALES[param]
    cm = matplotlib.colormaps[cmap]
    stops = np.linspace(0, 1, steps)
    return {
        "vmin": vmin,
        "vmax": vmax,
        "colors": ["#%02x%02x%02x" % tuple(int(255 * c) for c in cm(s)[:3]) for s in stops],
    }


def write_png(path: Path, rgba: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(path, rgba)


def time_key(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%MZ")


def write_manifest(outdir: Path, meta: dict, times: list[datetime], params: list[str]) -> Path:
    info = {**PARAMETERS, **DAILY_PARAMETERS}
    manifest = {
        **meta,
        "times": [time_key(t) for t in sorted(times)],
        "params": {
            p: {"units": info[p][0], "description": info[p][1], **legend(p)}
            for p in params
            if p in SCALES
        },
    }
    path = outdir / "manifest.json"
    path.write_text(json.dumps(manifest, indent=1))
    return path
