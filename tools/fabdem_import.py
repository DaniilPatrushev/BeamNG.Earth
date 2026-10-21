"""FABDEM into the server's local elevation (server/localdem.py; Moscow, 2026-10-21).

FABDEM V1-2 (Hawker et al. 2022, University of Bristol, CC BY-NC-SA 4.0: non-commercial use) is Copernicus GLO-30
with buildings and forests removed: a bare-earth model at 30 m. Download the 10 x 10 degree zips that cover your
region from https://data.bris.ac.uk/data/dataset/s5hqmjcdj8yo2ibzi9b4ew3sn (Moscow and its oblast:
N50E030-N60E040_FABDEM_V1-2.zip, and N50E040-N60E050_FABDEM_V1-2.zip for the oblast's eastern edge), then:

    python3 tools/fabdem_import.py ~/Downloads/N50E030-N60E040_FABDEM_V1-2.zip --bbox 54.2,35.1,56.98,40.25

Each 1 x 1 degree GeoTIFF (N55E037_FABDEM_V1-2.tif) becomes <cache>/dem/local/fabdem/N55E037.npy (float32, row 0
north, NaN for no data; ~35-52 MB each). Restart the server: it uses them ahead of Mapterhorn there (dem.Dem.local),
and the on-screen credits name FABDEM. Delete the folder (or set BNE_LOCAL_DEM=off) to go back.
"""
import argparse
import io
import os
import re
import sys
import zipfile

import numpy as np
from PIL import Image

NAME = re.compile(r"([NS])(\d{2})([EW])(\d{3})_FABDEM", re.I)
NODATA = -9999.0


def cell_of(name):
    m = NAME.search(os.path.basename(name))
    if not m:
        return None
    lat = int(m.group(2)) * (1 if m.group(1).upper() == "N" else -1)
    lon = int(m.group(4)) * (1 if m.group(3).upper() == "E" else -1)
    return lat, lon


def convert(data, out_path):
    """A FABDEM GeoTIFF's bytes to a float32 .npy (no data: NaN). Returns its shape."""
    Image.MAX_IMAGE_PIXELS = None
    try:
        a = np.asarray(Image.open(io.BytesIO(data)), np.float32)
    except Exception as e:          # a codec this Pillow lacks (FABDEM's COGs: deflate with a float predictor)
        try:
            import tifffile         # pip install tifffile imagecodecs
        except ImportError:
            raise SystemExit(f"Pillow cannot read this GeoTIFF ({e}); pip install tifffile imagecodecs") from None
        a = np.asarray(tifffile.imread(io.BytesIO(data)), np.float32)
    if a.ndim != 2:
        raise ValueError(f"unexpected image shape {a.shape}")
    a = np.where((a <= NODATA + 1) | ~np.isfinite(a), np.nan, a).astype(np.float32)
    tmp = out_path + ".tmp.npy"
    np.save(tmp, a)
    os.replace(tmp, out_path)
    return a.shape


def sources(paths):
    """(name, read()) of every FABDEM GeoTIFF in the given zips, folders and files."""
    for p in paths:
        if os.path.isdir(p):
            for root, _dirs, files in os.walk(p):
                for f in sorted(files):
                    full = os.path.join(root, f)
                    if f.lower().endswith(".zip"):
                        yield from sources([full])
                    elif f.lower().endswith(".tif"):
                        yield f, (lambda full=full: open(full, "rb").read())
        elif p.lower().endswith(".zip"):
            z = zipfile.ZipFile(p)
            for n in sorted(z.namelist()):
                if n.lower().endswith(".tif"):
                    yield n, (lambda n=n, z=z: z.read(n))
                elif n.lower().endswith(".zip"):      # (zips inside the zip)
                    inner = zipfile.ZipFile(io.BytesIO(z.read(n)))
                    for m in sorted(inner.namelist()):
                        if m.lower().endswith(".tif"):
                            yield m, (lambda m=m, inner=inner: inner.read(m))
        elif p.lower().endswith(".tif"):
            yield p, (lambda p=p: open(p, "rb").read())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="FABDEM zips, folders or GeoTIFFs")
    ap.add_argument("--bbox", help="lat0,lon0,lat1,lon1: only the 1 degree cells touching this box")
    ap.add_argument("--cache-dir", default=os.path.join(os.path.expanduser("~"), ".cache", "beamng-earth"))
    a = ap.parse_args(argv)
    box = [float(v) for v in a.bbox.split(",")] if a.bbox else None
    out_dir = os.path.join(a.cache_dir, "dem", "local", "fabdem")
    os.makedirs(out_dir, exist_ok=True)
    n = 0
    for name, read in sources(a.paths):
        c = cell_of(name)
        if c is None:
            continue
        lat, lon = c
        if box and not (lat + 1 > box[0] and lat < box[2] and lon + 1 > box[1] and lon < box[3]):
            continue
        out = os.path.join(out_dir, f"{'N' if lat >= 0 else 'S'}{abs(lat):02d}{'E' if lon >= 0 else 'W'}{abs(lon):03d}.npy")
        shape = convert(read(), out)
        n += 1
        print(f"{os.path.basename(name)} -> {out} {shape}")
    if not n:
        print("no FABDEM tiles found (in the box)", file=sys.stderr)
        return 1
    print(f"{n} cells in {out_dir}; restart the server to use them")
    return 0


if __name__ == "__main__":
    sys.exit(main())
