# WRF + WPS (geogrid/metgrid only) + raspwn.
#
#   docker build -f docker/Dockerfile -t raspwn .
#
# ungrib is not used: raspwn writes WPS intermediate files itself, so WPS
# is built without GRIB2 support (no jasper/libpng dance).

ARG UBUNTU=24.04
ARG WRF_VERSION=v4.6.1
ARG WPS_VERSION=v4.6.0

FROM ubuntu:${UBUNTU} AS build
ARG WRF_VERSION
ARG WPS_VERSION
ARG JOBS=8
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gfortran m4 perl csh file git ca-certificates \
        libnetcdf-dev libnetcdff-dev mpich libmpich-dev zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# WRF's configure expects a single $NETCDF prefix with include/ and lib/
RUN mkdir -p /opt/netcdf \
    && ln -s /usr/include /opt/netcdf/include \
    && ln -s /usr/lib/$(uname -m)-linux-gnu /opt/netcdf/lib
ENV NETCDF=/opt/netcdf \
    WRF_EM_CORE=1 \
    WRF_NMM_CORE=0 \
    WRFIO_NCD_LARGE_FILE_SUPPORT=1

WORKDIR /opt
RUN git clone --depth 1 --branch ${WRF_VERSION} --recurse-submodules https://github.com/wrf-model/WRF.git
# 34 = GNU gfortran/gcc dmpar, 1 = basic nesting.
# Ubuntu's MPICH wrappers inject -flto=auto, which breaks WRF's link step
# (undefined module_optional_input symbols): turn LTO back off.
RUN cd WRF && printf '34\n1\n' | ./configure \
    && sed -i -e 's/^DM_FC *=.*/& -fno-lto/' -e 's/^DM_CC *=.*/& -fno-lto/' configure.wrf \
    && ./compile -j ${JOBS} em_real > compile.log 2>&1; \
    tail -20 compile.log; test -x main/wrf.exe && test -x main/real.exe

RUN git clone --depth 1 --branch ${WPS_VERSION} https://github.com/wrf-model/WPS.git
# 2 = gfortran serial, no GRIB2
RUN cd WPS && printf '2\n' | WRF_DIR=/opt/WRF ./configure \
    && ./compile > compile.log 2>&1; \
    tail -20 compile.log; test -x geogrid/src/geogrid.exe && test -x metgrid/src/metgrid.exe


FROM ubuntu:${UBUNTU}
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnetcdf19t64 libnetcdff7 mpich libgfortran5 \
        python3 python3-venv ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/WRF/run /opt/WRF/run
COPY --from=build /opt/WRF/main/wrf.exe /opt/WRF/main/real.exe /opt/WRF/main/
COPY --from=build /opt/WPS/geogrid/GEOGRID.TBL.ARW /opt/WPS/geogrid/
COPY --from=build /opt/WPS/metgrid/METGRID.TBL.ARW /opt/WPS/metgrid/
COPY --from=build /opt/WPS/geogrid/src/geogrid.exe /opt/WPS/metgrid/src/metgrid.exe /opt/WPS/
# run/ contains symlinks to ../main/*.exe; make them real paths
RUN ln -sf /opt/WRF/main/wrf.exe /opt/WRF/run/wrf.exe \
    && ln -sf /opt/WRF/main/real.exe /opt/WRF/run/real.exe

COPY pyproject.toml README.md /src/
COPY src /src/src
RUN python3 -m venv /opt/raspwn && /opt/raspwn/bin/pip install --no-cache-dir /src
ENV PATH=/opt/raspwn/bin:$PATH \
    MPLCONFIGDIR=/tmp

COPY regions /regions
VOLUME ["/data"]
ENTRYPOINT ["raspwn"]
