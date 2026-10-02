#!/usr/bin/env bash
# End-to-end smoke test without WeatherNext access:
# synthetic WeatherNext-shaped Zarr + live GFS soil -> geogrid, metgrid,
# real, a 3 h WRF run at 12 km, and the soaring maps.
#
#   scripts/smoke/run.sh /path/to/workdir      (needs the raspwn image)
set -euo pipefail
work=$(realpath "${1:?usage: run.sh WORKDIR}")
here=$(dirname "$(realpath "$0")")
mkdir -p "$work"
cp "$here/region.yaml" "$here/make_store.py" "$work/"
if [ ! -d "$work/geog/topo_gmted2010_5m" ]; then
  mkdir -p "$work/geog"
  curl -fL https://www2.mmm.ucar.edu/wrf/src/wps_files/geog_low_res_mandatory.tar.gz \
    | tar -xz -C "$work/geog" --strip-components=1
fi
docker run --rm -v "$work":/smoke --entrypoint python3 raspwn /smoke/make_store.py /smoke/wn.zarr >/dev/null
docker run --rm -v "$work":/smoke raspwn run -c /smoke/region.yaml
echo "maps in $work/out (python3 -m http.server -d $work/out)"
