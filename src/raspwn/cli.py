"""Command line interface: ``raspwn <command> -c region.yaml``."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from . import config as cfgmod
from . import pipeline
from .wnext import source

STEPS = ["geogrid", "fetch", "metgrid", "wrf", "post"]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="raspwn", description=__doc__)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("-c", "--config", required=True, type=Path)
        p.add_argument("--init", help="init time, e.g. 2026-10-02T00 (default: config)")
        p.add_argument("--member", help="ensemble member index or 'mean'")

    p = sub.add_parser("inspect", help="print the WeatherNext store schema")
    common(p)

    p = sub.add_parser("run", help="run the full pipeline or selected steps")
    common(p)
    p.add_argument("--day", type=int, action="append", help="forecast day(s), default: config")
    p.add_argument("--steps", default=",".join(STEPS), help=f"comma list from {STEPS}")

    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    cfg = cfgmod.load(args.config)
    if args.member is not None:
        cfg.weathernext.member = args.member if args.member == "mean" else int(args.member)
    init = pipeline.resolve_init(cfg, args.init)

    if args.cmd == "inspect":
        print(f"init {init:%Y-%m-%d %HZ}")
        print(source.describe(cfg.weathernext, init))
        return

    steps = [s.strip() for s in args.steps.split(",")]
    unknown = set(steps) - set(STEPS)
    if unknown:
        ap.error(f"unknown steps {sorted(unknown)}")
    for day in args.day or cfg.forecast.days:
        run = pipeline.Run(cfg, init, day)
        for step in STEPS:
            if step in steps:
                getattr(run, step)()
        logging.info("day %d done: %s", day, run.dir)


if __name__ == "__main__":
    main()
