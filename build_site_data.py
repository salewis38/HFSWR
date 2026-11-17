#!/usr/bin/env python3
"""
build_site_data.py
==================
Extracts the site-specific ITU-R P.372 atmospheric-noise coefficients and the
ITU-R P.1239 (CCIR) monthly-median foF2 values for one receiver location from
the official ITU-R Study Group 3 data files, and writes them to a compact JSON
file read by hfswr_tool.py.

Source data: https://github.com/ITU-R-Study-Group-3/ITU-R-HF  (folder P372/Data)
  COEFFmmW.txt  - harmonised monthly coefficient files (fakp, fakabp, dud, fam)
  ionosmm.bin   - 1.5-degree foF2 and M(3000)F2 grids for R12 = 0 and 100

The ITU states this software/data "may be used by implementers ... free from
any copyright assertions".

Usage:
    python build_site_data.py <path-to-ITU-R-HF/P372/Data> <lat_deg> <lon_deg> <out.json> [site name]
Example (Land's End):
    python build_site_data.py ITU-R-HF/P372/Data 50.066 -5.715 site_lands_end.json "Land's End"
"""
import json
import math
import os
import sys

import numpy as np


# --------------------------------------------------------------------------
# Coefficient file parsing
# --------------------------------------------------------------------------
def _read_block(lines, header, count):
    """Return `count` floats that follow the line `header`."""
    idx = next(i for i, l in enumerate(lines) if l.strip() == header)
    vals = []
    i = idx + 1
    while len(vals) < count:
        vals.extend(float(v) for v in lines[i].split())
        i += 1
    return np.array(vals[:count])


def read_coeff_file(path):
    with open(path) as fh:
        lines = fh.readlines()
    # C-order reshapes exactly as in ITU P372 Noise.c ReadFamDud()
    fakp = _read_block(lines, "fakp(29,16,6)", 29 * 16 * 6).reshape(6, 16, 29)
    fakabp = _read_block(lines, "fakabp(2,6)", 12).reshape(6, 2)
    dud = _read_block(lines, "dud(5,12,5)", 300).reshape(5, 12, 5)
    fam = _read_block(lines, "fam(14,12)", 168).reshape(12, 14)
    return fakp, fakabp, dud, fam


def fam_1mhz(fakp, fakabp, tmblk, lat_rad, lon_rad):
    """Fam at 1 MHz from the Fourier series (port of GetFamParameters())."""
    q = (lon_rad + 2 * math.pi) / 2 if lon_rad < 0 else lon_rad / 2
    k = np.arange(1, 16)
    zz = np.array([np.sum(np.sin(k * q) * fakp[tmblk, :15, j]) + fakp[tmblk, 15, j]
                   for j in range(29)])
    q = lat_rad + math.pi / 2
    r = np.sum(np.sin(np.arange(1, 30) * q) * zz)
    return float(r + fakabp[tmblk, 0] + fakabp[tmblk, 1] * q)


# --------------------------------------------------------------------------
# foF2 grid
# --------------------------------------------------------------------------
def read_fof2_site(path, lat_deg, lon_deg):
    """Bilinear foF2 and M(3000)F2 at the site, 24 UTC hours, R12 = 0 and 100."""
    nh, nlon, nlat, nssn = 24, 241, 121, 2
    n = nh * nlon * nlat * nssn
    with open(path, "rb") as fh:
        fh.read(5)
        fo = np.frombuffer(fh.read(4 * n), dtype="<f4")
        fh.read(10)
        m3 = np.frombuffer(fh.read(4 * n), dtype="<f4")
    return (_site_interp(fo, lat_deg, lon_deg), _site_interp(m3, lat_deg, lon_deg))


def _site_interp(buf, lat_deg, lon_deg):
    nh, nlon, nlat, nssn = 24, 241, 121, 2
    grid = buf.reshape(nssn, nlon, nlat, nh)          # [ssn][lon][lat][hour]
    x = (lon_deg + 180.0) / 1.5
    y = (lat_deg + 90.0) / 1.5
    j0, k0 = int(math.floor(x)), int(math.floor(y))
    fx, fy = x - j0, y - k0
    g = (grid[:, j0, k0, :] * (1 - fx) * (1 - fy) + grid[:, j0 + 1, k0, :] * fx * (1 - fy)
         + grid[:, j0, k0 + 1, :] * (1 - fx) * fy + grid[:, j0 + 1, k0 + 1, :] * fx * fy)
    # P.533 convention: grid hour index n corresponds to (n+1) UTC.
    # Re-order so that element h is h UTC (h = 0..23).
    g = np.roll(g, 1, axis=1)
    return g  # shape (2, 24): [R12=0, R12=100][UTC hour]


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(1)
    data_dir, lat, lon, out = sys.argv[1], float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    name = sys.argv[5] if len(sys.argv) > 5 else f"{lat:.3f},{lon:.3f}"
    lat_r, lon_r = math.radians(lat), math.radians(lon)
    hemi = 0 if lat >= 0 else 6

    site = {"name": name, "lat": lat, "lon": lon, "months": []}
    for m in range(1, 13):
        fakp, fakabp, dud, fam = read_coeff_file(os.path.join(data_dir, f"COEFF{m:02d}W.txt"))
        fof2, m3k = read_fof2_site(os.path.join(data_dir, f"ionos{m:02d}.bin"), lat, lon)
        site["months"].append({
            "fam1": [fam_1mhz(fakp, fakabp, tb, lat_r, lon_r) for tb in range(6)],
            "fam": fam[hemi:hemi + 6].tolist(),                  # [tmblk][14]
            "dud": dud[:, hemi:hemi + 6, :].tolist(),            # [5][tmblk][5]
            "fof2_r0": [round(float(v), 3) for v in fof2[0]],
            "fof2_r100": [round(float(v), 3) for v in fof2[1]],
            "m3k_r0": [round(float(v), 4) for v in m3k[0]],
            "m3k_r100": [round(float(v), 4) for v in m3k[1]],
        })
    with open(out, "w") as fh:
        json.dump(site, fh, separators=(",", ":"))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
