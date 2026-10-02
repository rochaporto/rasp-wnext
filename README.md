# rasp-wnext

Open-source soaring forecasts in the style of RASP/BLIPMAP (and of commercial
services like SkySight or TopMeteo), using **Google WeatherNext** instead of
GFS/ECMWF to drive a high-resolution WRF model.

```
 WeatherNext 3 (Zarr on GCS)          GFS f0xx (AWS open data)
 13 p-levels @0.25°, sfc @0.1°        soil T/moisture, skin T, snow,
 hourly, 64-member ensemble           land mask, terrain  (initial time)
            │                                   │
            ▼  raspwn fetch                     ▼
     WNEXT:YYYY-MM-DD_HH            GFSLSM:YYYY-MM-DD_HH      ← WPS intermediate
            └───────────────┬───────────────────┘              files written in
                            ▼                                   Python, no ungrib
                metgrid.exe  →  real.exe  →  wrf.exe   (9 km → 3 km nest)
                                                 │
                                                 ▼  raspwn post
              W*, Hcrit, cu/OD cloudbase, B/S ratio, stars, flight distance…
                       PNG overlays + manifest.json + Leaflet viewer
```

## Why it is built this way

**The RASP structure is kept, the code is new.** A RASP run is WPS → WRF → a
post-processor computing BLIPMAP parameters. The original post-processing
relies on NCL (deprecated) and on DrJack's Fortran routines, which only
circulate as compiled `.so` binaries. This project keeps the pipeline but
reimplements the parameters in Python/numpy (`src/raspwn/blip/params.py`).

**WeatherNext replaces GFS as first guess and boundary conditions.** Python
reads WeatherNext Zarr directly and writes WPS *intermediate format*
files, so GRIB and `ungrib.exe` are no longer involved. Any other source
can be added the same way.

**GFS fills in the land surface.** WeatherNext only predicts the atmosphere
and a few surface variables. WRF's land surface model still needs soil
temperature and moisture, skin temperature and snow at the start time. For
those, ~16 GRIB messages are pulled from GFS through HTTP byte ranges and
passed to metgrid as `constants_name`. Surface pressure is recomputed from the
WeatherNext column at GFS terrain height, so it stays consistent with the
WeatherNext atmosphere.

**Thermals come from WRF, not from WeatherNext.** WeatherNext has no boundary
layer, surface fluxes or 3 km terrain. Those make the thermals, and WRF
produces them during the run. WeatherNext supplies the large-scale state:
airmass, synoptic wind, moisture, and lapse rates aloft. This is exactly
GFS's role in classic RASP.

## WeatherNext access

1. Request access with the WeatherNext data request form
   (<https://developers.google.com/weathernext/guides/access-forecast>).
   Approval usually takes about a week and is free.
2. The full-ensemble bucket `gs://weathernext3_spatial` is **requester
   pays**. Set `weathernext.billing_project` to a GCP project you own. A
   regional subset is small (a few MB per run). Egress is cheapest from
   `us-east1`.
3. Credentials: `gcloud auth application-default login`, or set
   `GOOGLE_APPLICATION_CREDENTIALS` to a service account key. Mount it into
   the container.
4. **Licence:** data that is less than 1 h old falls under the *GDM
   Real-Time Weather Forecasting Experimental Data Terms of Use*. Older data
   is CC BY 4.0. Read the terms before you publish forecasts.

The store schema isn't formally versioned, so check it before your first
run:

```
raspwn inspect -c regions/example/region.yaml --init 2026-10-01T00
```

Variable, dimension and level names are all configurable under
`weathernext:` (see `WeatherNextConfig` in `src/raspwn/wnext/source.py`).
Both layouts are supported: one variable per level (`temperature_850`) and
variables with a level dimension.

## Running

```bash
docker build -f docker/Dockerfile -t raspwn .
scripts/get-geog.sh /data/geog                 # once, ~29 GB

docker run --rm -v /data:/data -v ~/.config/gcloud:/root/.config/gcloud \
    -v $PWD/regions:/regions raspwn run -c /regions/example/region.yaml
# or step by step: --steps geogrid,fetch  then  --steps metgrid,wrf,post
# options: --init 2026-10-02T00  --day 1  --member mean

python3 -m http.server -d /data/out 8000          # open http://localhost:8000
```

To add a region, copy `regions/example/`. Then edit the Lambert projection
and the domains (outer domain about 9 km, nest about 3 km, like RASP), and
the forecast window (`start_offset_hours`, `length_hours`, `days`).
`wrf.physics` overrides namelist physics. `wrf.templates` points to your own
`namelist.*.j2` files.

## Parameters

| name | meaning |
|---|---|
| `wstar` | Deardorff convective velocity from WRF surface buoyancy flux and BL depth |
| `hwcrit` / `dwcrit` | height MSL / depth AGL where updraft falls below 225 fpm |
| `hbl` / `dbl` | boundary-layer top MSL / depth (WRF `PBLH`) |
| `zsfclcl`, `zsfclcldif` | cumulus cloudbase (surface parcel LCL) and cu potential |
| `zblcl`, `zblcldif` | overdevelopment cloudbase (BL-mean moisture) and OD potential |
| `hglider` / `dglider` | usable thermalling height = min(Hcrit, cu base, OD base) |
| `bsratio` | buoyancy/shear ratio W*/(√Cd·U_bl) |
| `stars` | RASP 0–5 rating (UK-tuned heuristic) |
| `pfd`, `pfd_tot` | MacCready cross-country speed per hour and its daily sum, in km (potential flight distance) |
| `blwind`, `bltopwind`, `blwindshear`, `sfcwind` | winds |
| `sfctemp`, `sfcdewpt`, `sfcsun`, `sfcshf`, `rain1` | surface fields |
| `blcloud`, `cfracl/m/h` | cloud fractions |

## Status and caveats

* **Tested:** the unit and integration tests (`pytest`) cover these parts:
  the intermediate file format, byte for byte; WeatherNext normalisation on
  a synthetic store shaped like WN3, including bounding boxes that cross
  the Greenwich meridian, the 6 h + sub-hour lead times, member and mean
  selection; surface pressure against ISA; GRIB decoding; namelist
  rendering; the parameter formulas on idealised columns; Lambert →
  Web-Mercator resampling; and post-processing from a synthetic `wrfout`
  to PNGs.
* **End-to-end smoke test** (`scripts/smoke/run.sh`), run successfully
  in the Docker image. It uses a synthetic WeatherNext-shaped store and
  live GFS soil data, then runs geogrid → metgrid → real → 3 h of WRF at
  12 km over Portugal → maps. Results are physically plausible: W* about
  1.5 m/s and a BL of about 800 m at 10:00 local in October. This
  validates the intermediate files, the GFS/WeatherNext merge, the
  namelists and post-processing against real WPS/WRF binaries.
* **Not yet tested against live data:** WeatherNext access needs an
  approved account. The real store's dimension and variable names may need
  config tweaks; `raspwn inspect` shows them.
* **Hcrit's thermal profile** is Allen (2006), scaled so its peak equals
  W*. DrJack's exact profile is not public. Calibrate it against observed
  climbs or an existing RASP before you trust absolute Hcrit values.
* **Vertical input resolution:** WeatherNext has 13 pressure levels, with
  only 1000/925/850/700 hPa below 3 km. That is coarser than GFS's levels.
  WRF builds its own boundary layer within a few hours, which is why the run
  starts at 06Z for a European day. Keep the spin-up.
* WeatherNext doesn't provide cloud water/ice or soil state. Clouds spin up
  in WRF, and soil comes from GFS.

### Build notes

* Ubuntu 24.04's MPICH wrappers add `-flto=auto`, which breaks WRF's
  link step. The Dockerfile appends `-fno-lto`.
* `real.exe` runs single-process. With several MPI ranks, this toolchain
  writes a corrupted first `wrfbdy` time. `wrf.exe` uses `wrf.mpi_procs`.
* Thompson microphysics builds lookup tables on first use, which takes
  about 5 minutes. They are cached in `<workdir>/cache`.

## Ideas / roadmap

* Run several ensemble members (`--member 0..N`) and publish exceedance
  probabilities, for example P(W* > 1.5 m/s). This is something no
  GFS-based RASP can do cheaply.
* Skew-T soundings and cross-sections at sites, plus GeoTIFF/COG export.
* Schedule with cron or systemd after each 00Z WeatherNext init.

## Credits

The parameter definitions follow Dr. John W. (Jack) Glendening's
BLIPMAP/RASP. The RASP NCL drivers in
[sfalmo/rasp-from-scratch](https://github.com/sfalmo/rasp-from-scratch)
were a reference for vhf, B/S ratio, stars and pfd. WRF/WPS are from
NCAR/MMM. WeatherNext is from Google DeepMind/Google Research.
