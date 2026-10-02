#!/usr/bin/env bash
# Download the WPS static geographical data (terrain, land use, soil...).
# The "mandatory high-res" set (~2.6 GB download, ~29 GB unpacked) is what
# geog_data_res = 'default' uses.
set -euo pipefail
dest=${1:-/data/geog}
url=https://www2.mmm.ucar.edu/wrf/src/wps_files/geog_high_res_mandatory.tar.gz
mkdir -p "$dest"
curl -fL "$url" | tar -xz -C "$dest" --strip-components=1
echo "WPS geog data in $dest"
