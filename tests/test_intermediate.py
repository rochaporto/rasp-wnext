import struct
from datetime import datetime

import numpy as np

from raspwn.wps.intermediate import MISSING, Field, file_name, read_file, write_file


def test_roundtrip(tmp_path):
    data = np.arange(12, dtype=np.float32).reshape(3, 4)
    data[0, 0] = np.nan
    f = Field("TT", "K", "Temperature", 85000.0, datetime(2026, 10, 2, 6),
              startlat=38.0, startlon=-10.0, dlat=0.25, dlon=0.25, data=data)
    path = write_file(tmp_path / file_name("WNEXT", f.date), [f, f])
    assert path.name == "WNEXT:2026-10-02_06"

    back = read_file(path)
    assert len(back) == 2
    b = back[0]
    assert (b.name, b.units, b.level, b.nx, b.ny) == ("TT", "K", 85000.0, 4, 3)
    assert b.date == f.date
    assert np.isclose(b.startlon, -10.0) and np.isclose(b.dlat, 0.25)
    assert b.data[0, 0] == np.float32(MISSING)
    np.testing.assert_array_equal(b.data.ravel()[1:], data.ravel()[1:])


def test_binary_layout_matches_fortran(tmp_path):
    """Big-endian records with 4-byte length markers, x varying fastest."""
    data = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)  # ny=2, nx=3
    f = Field("GHT", "m", "Height", 50000.0, datetime(2026, 1, 1),
              startlat=0, startlon=0, dlat=1, dlon=1, data=data)
    raw = write_file(tmp_path / "x", [f]).read_bytes()

    assert struct.unpack(">iii", raw[:12]) == (4, 5, 4)
    (hdr_len,) = struct.unpack(">i", raw[12:16])
    assert hdr_len == 24 + 4 + 32 + 9 + 25 + 46 + 4 + 12  # = 156
    nx, ny, iproj = struct.unpack(">3i", raw[16 + 144 : 16 + 156])
    assert (nx, ny, iproj) == (3, 2, 0)
    slab = raw[-(4 + 24) : -4]
    assert struct.unpack(">6f", slab) == (1, 2, 3, 4, 5, 6)
