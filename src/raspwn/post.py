"""Post-processing: wrfout files -> PNG overlays + manifest for the viewer."""

from __future__ import annotations

import json
import logging
import re
import shutil
from datetime import datetime
from pathlib import Path

from .blip import render, wrfout
from .config import Config

log = logging.getLogger(__name__)

_WRFOUT = re.compile(r"wrfout_d(\d\d)_(\d{4}-\d\d-\d\d_\d\d:\d\d:\d\d)")


def wrfout_files(run_dir: Path, domain: int | None) -> list[Path]:
    files = [p for p in run_dir.glob("wrfout_d*") if _WRFOUT.match(p.name)]
    if not files:
        raise FileNotFoundError(f"no wrfout files in {run_dir}")
    doms = sorted({int(_WRFOUT.match(p.name).group(1)) for p in files})
    dom = domain or doms[-1]
    return sorted(p for p in files if int(_WRFOUT.match(p.name).group(1)) == dom)


def process(cfg: Config, run_dir: Path, init: datetime, day: int) -> Path:
    member = cfg.weathernext.member
    outdir = Path(cfg.output.dir) / cfg.region.name / f"{init:%Y%m%d%H}_d{day}_m{member}"
    wanted = [p for p in cfg.output.params if p in render.SCALES]

    resampler = None
    prev_rain = None
    pfd_tot = None
    times = []
    for path in wrfout_files(run_dir, cfg.output.domain):
        state, grid, valid = wrfout.read_state(path)
        fields = wrfout.compute(state, prev_rain)
        prev_rain = state["rain"]
        if resampler is None:
            resampler = render.Resampler(grid)
        times.append(valid)
        key = render.time_key(valid)
        for p in wanted:
            if p in fields:
                render.write_png(outdir / p / f"{key}.png", render.colourise(resampler(fields[p]), p))
        pfd_tot = fields["pfd"] if pfd_tot is None else pfd_tot + fields["pfd"]
        log.info("post %s done", key)

    if "pfd_tot" in wanted and pfd_tot is not None:
        # one map for the whole day, shown at every time step
        rgba = render.colourise(resampler(pfd_tot), "pfd_tot")
        for t in times:
            render.write_png(outdir / "pfd_tot" / f"{render.time_key(t)}.png", rgba)

    meta = dict(
        region=cfg.region.name,
        init=init.strftime("%Y-%m-%dT%HZ"),
        model="WRF driven by Google WeatherNext",
        member=member,
        bounds=resampler.bounds,
    )
    render.write_manifest(outdir, meta, times, wanted)
    _update_index(Path(cfg.output.dir), outdir)
    return outdir


def _update_index(root: Path, run_out: Path) -> None:
    """runs.json at the output root lists every run; index.html shows them."""
    shutil.copy(Path(__file__).parent / "web" / "index.html", root / "index.html")
    index = root / "runs.json"
    runs = json.loads(index.read_text()) if index.exists() else []
    rel = str(run_out.relative_to(root))
    runs = sorted(set(runs) | {rel})
    index.write_text(json.dumps(runs, indent=1))

