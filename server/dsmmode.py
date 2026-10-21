"""DSM mode: the places whose elevation is a 30 m surface model (Copernicus GLO-30 through dsmground), not lidar
(Moscow, 2026-10-21: no Mapterhorn tile deeper than z12 over the city).

The road and bridge constants were tuned on national lidar (the US, western Europe). On a 30 m surface model a road's
profile carries the model's residual bumps (blocks and trees partly filtered, 1-2 m noise, a bilinear kink every pixel)
at 30-150 m wavelengths, and the ground filter's opening (dsmground.WINDOW_M) erases every embankment narrower than
its window, overpass approaches included. The generators read is_dsm() and switch to settings for such ground: longer
profile smoothing, raised approaches for every road class, no retaining walls.

    dsmmode.register(dem)           # the process's dem.Dem (earthd, the block workers)
    dsmmode.is_dsm(lat, lon)        # per z12 tile: no lidar under it

BNE_DSM_MODE=auto (default) asks the registered Dem (dem.Dem._lidar: two z13 tiles, cached with the DEM tiles);
on / off force it everywhere. Without a registered Dem (tests, tools) auto is off: every result stays as before.
A tile whose check failed (offline, nothing cached) keeps the lidar settings and is asked again next time.
"""
import math
import os
import threading

GLOBAL_ZOOM = 12
MAX_LAT = 85.0511287798

_dem = None
_cache = {}
_lock = threading.Lock()


def mode():
    m = os.environ.get("BNE_DSM_MODE", "auto").strip().lower()
    return m if m in ("on", "off") else "auto"


def register(dem):
    """The Dem that answers is_dsm in auto mode (None: none)."""
    global _dem
    with _lock:
        _dem = dem
        _cache.clear()


def tile_of(lat, lon):
    n = 2 ** GLOBAL_ZOOM
    lat = max(-MAX_LAT, min(MAX_LAT, float(lat)))
    x = int((float(lon) / 360.0 + 0.5) * n) % n
    y = int((0.5 - math.asinh(math.tan(math.radians(lat))) / (2 * math.pi)) * n)
    return x, min(max(y, 0), n - 1)


def is_dsm(lat, lon):
    """Is the elevation at (lat, lon) a surface model (no lidar)? Deterministic per z12 tile."""
    m = mode()
    if m != "auto":
        return m == "on"
    dem = _dem
    if dem is None:
        return False
    key = tile_of(lat, lon)
    with _lock:
        got = _cache.get(key)
    if got is not None:
        return got
    import netfetch
    ok = True
    lidar = False
    try:
        for cx, cy in ((2 * key[0], 2 * key[1]), (2 * key[0] + 1, 2 * key[1] + 1)):
            if dem._download(_mapterhorn(), GLOBAL_ZOOM + 1, cx, cy) is not None:
                lidar = True
                break
    except netfetch.FetchError:
        ok = False
    except Exception:
        ok = False
    if not ok:
        return False
    res = not lidar
    if ok:
        with _lock:
            _cache[key] = res
    return res


def _mapterhorn():
    import dem as demmod
    return demmod.MAPTERHORN
