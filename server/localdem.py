"""Local elevation tiles ahead of Mapterhorn: a bare-earth model the user downloaded for a region (Moscow, 2026-10-21:
FABDEM, Copernicus GLO-30 with buildings and forests removed by machine learning; Hawker et al. 2022, University of
Bristol, CC BY-NC-SA 4.0).

    <cache>/dem/local/<source>/N55E037.npy      float32 [rows, cols], one 1 x 1 degree cell, row 0 north,
                                                 pixel-is-area (rows / cols pixels per degree; NaN = no data)

tools/fabdem_import.py writes them from FABDEM's GeoTIFFs. dem.Dem reads them through LocalDem: a 512 px tile with
every pixel inside the local cells is made from them alone (no ground filter: they are a terrain model already);
one partly inside is the local heights blended into Mapterhorn's over FEATHER_PX at the coverage's edge. Both are
EGM2008 heights (FABDEM's vertical datum is Copernicus's), so the edge needs no offset.
"""
import glob
import hashlib
import math
import os
import re
import threading

import numpy as np

FEATHER_PX = 48             # px of a 512 px tile over which the local heights fade into Mapterhorn's at their edge
MIN_ZOOM = 10               # coarser tiles stay Mapterhorn's (each spans degrees: far tiles, where 30 m changes nothing)
NAME = re.compile(r"^([NS])(\d{1,2})([EW])(\d{1,3})\.npy$")
SHORT = "FABDEM"
CREDIT = "FABDEM V1-2 (Hawker et al. 2022, University of Bristol; CC BY-NC-SA 4.0)"


def cell_name(lat0, lon0):
    """The file name of the 1 degree cell whose south-west corner is (lat0, lon0) (integers)."""
    return f"{'N' if lat0 >= 0 else 'S'}{abs(lat0):02d}{'E' if lon0 >= 0 else 'W'}{abs(lon0):03d}.npy"


class LocalDem:
    """The local 1 degree cells under a folder (every source subfolder; the first found wins for a cell)."""

    def __init__(self, root):
        self.root = root
        self.cells = {}             # (lat0, lon0) -> path
        self.source = None
        self._mm = {}
        self._lock = threading.Lock()
        for path in sorted(glob.glob(os.path.join(root, "*", "*.npy"))):
            m = NAME.match(os.path.basename(path))
            if not m:
                continue
            lat0 = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
            lon0 = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
            self.cells.setdefault((lat0, lon0), path)
            self.source = self.source or os.path.basename(os.path.dirname(path))
        h = hashlib.blake2b(digest_size=6)
        for k in sorted(self.cells):
            p = self.cells[k]
            h.update(f"{p}:{os.path.getmtime(p)}".encode())
        self.tag = h.hexdigest()

    def __bool__(self):
        return bool(self.cells)

    def _arr(self, key):
        with self._lock:
            a = self._mm.get(key)
            if a is None:
                a = np.load(self.cells[key], mmap_mode="r", allow_pickle=False)
                self._mm[key] = a
            return a

    def covers_tile(self, z, x, y):
        """Do any local cells overlap tile (z, x, y) (z >= MIN_ZOOM)?"""
        if z < MIN_ZOOM or not self.cells:
            return False
        lat_n, lon_w, lat_s, lon_e = tile_bounds(z, x, y)
        for la in range(math.floor(lat_s), math.floor(lat_n) + 1):
            for lo in range(math.floor(lon_w), math.floor(lon_e) + 1):
                if (la, lo) in self.cells:
                    return True
        return False

    def sample(self, lat, lon):
        """Bilinear heights at lat / lon arrays (same shape), NaN outside the local cells. Pure given the files."""
        lat = np.asarray(lat, np.float64)
        lon = np.asarray(lon, np.float64)
        out = np.full(lat.shape, np.nan, np.float64)
        la0 = np.floor(lat).astype(np.int64)
        lo0 = np.floor(lon).astype(np.int64)
        for key in set(zip(la0.ravel().tolist(), lo0.ravel().tolist())):
            if key not in self.cells:
                continue
            m = (la0 == key[0]) & (lo0 == key[1])
            a = self._arr(key)
            rows, cols = a.shape
            # pixel centres at (r + 0.5) / rows south of the cell's north edge, (c + 0.5) / cols east of its west
            fr = (key[0] + 1 - lat[m]) * rows - 0.5
            fc = (lon[m] - key[1]) * cols - 0.5
            r0 = np.clip(np.floor(fr).astype(np.int64), 0, rows - 2)
            c0 = np.clip(np.floor(fc).astype(np.int64), 0, cols - 2)
            wr = np.clip(fr - r0, 0.0, 1.0)
            wc = np.clip(fc - c0, 0.0, 1.0)
            v00, v01 = a[r0, c0], a[r0, c0 + 1]
            v10, v11 = a[r0 + 1, c0], a[r0 + 1, c0 + 1]
            out[m] = (v00 * (1 - wc) + v01 * wc) * (1 - wr) + (v10 * (1 - wc) + v11 * wc) * wr
        return out

    def tile(self, z, x, y, size):
        """(heights float32 [size, size] (NaN outside), weight float32 [size, size]: 1 well inside the local cells,
        falling to 0 over FEATHER_PX toward their edge) of Web Mercator tile (z, x, y), pixel centres as dem.py's."""
        n = 2 ** z
        i = (np.arange(size) + 0.5) / size
        lon = (x + i) / n * 360.0 - 180.0
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + i) / n))))
        LA, LO = np.meshgrid(lat, lon, indexing="ij")
        h = self.sample(LA, LO).astype(np.float32)
        ok = np.isfinite(h)
        if ok.all():
            return h, np.ones((size, size), np.float32)
        if not ok.any():
            return h, np.zeros((size, size), np.float32)
        # distance (px) to the nearest pixel without local data, by a chamfer pass, capped at FEATHER_PX
        d = np.where(ok, float(FEATHER_PX), 0.0).astype(np.float32)
        for _ in range(2):
            for r in range(1, size):
                d[r] = np.minimum(d[r], d[r - 1] + 1)
            for r in range(size - 2, -1, -1):
                d[r] = np.minimum(d[r], d[r + 1] + 1)
            for c in range(1, size):
                d[:, c] = np.minimum(d[:, c], d[:, c - 1] + 1)
            for c in range(size - 2, -1, -1):
                d[:, c] = np.minimum(d[:, c], d[:, c + 1] + 1)
        u = np.clip(d / FEATHER_PX, 0.0, 1.0)
        return h, (u * u * (3 - 2 * u)).astype(np.float32)


def tile_bounds(z, x, y):
    """(lat north, lon west, lat south, lon east) of Web Mercator tile (z, x, y)."""
    n = 2 ** z
    lon_w = x / n * 360.0 - 180.0
    lon_e = (x + 1) / n * 360.0 - 180.0
    lat_n = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    lat_s = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return lat_n, lon_w, lat_s, lon_e
