"""Road and bridge audit of a place, offline from the game (Moscow DSM work, 2026-10-21).

Builds what the server builds for the physics ground around a point (the DEM, the road grading, the bridge decks)
and prints:
  roads:   per class, the rms of the graded centreline's 2nd difference per 10 m (bumpiness), its p95 and max grade
  joins:   the height step where two ground roads meet (an end on another road), p95 / max
  decks:   per bridge part crossing a ground road, the deck's height over that road (clearance), and how many sit
           under 4.5 m; the steepest grade from the road onto a deck end

    python3 tools/road_audit.py --at 55.7905,37.5310 --radius 700            # one run, BNE_DSM_MODE as set
    python3 tools/road_audit.py --at 55.7905,37.5310 --compare               # off vs on, side by side

Uses the server's cache (~/.cache/beamng-earth) and the network for what it lacks.
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))


def _pct(a, q):
    return float(np.percentile(a, q)) if len(a) else float("nan")


def audit(lat, lon, radius, cache_dir):
    import dem as demmod
    import dsmmode
    import geodesy
    import grading
    import mapcells
    import mapdata as mapdatamod
    import netfetch
    import roads as roadsmod

    net = netfetch.Fetcher()
    dem = demmod.Dem(os.path.join(cache_dir, "dem"), fetcher=net)
    dsmmode.register(dem)
    md = mapdatamod.MapData(cache_dir, fetcher=net)
    frame = geodesy.EnuFrame(lat, lon)

    def raw(la, lo):
        return dem.sample(la, lo, grading.PROFILE_ZOOM)
    g = grading.Grader(frame, md, raw)

    def elev(la, lo):
        return g.apply(la, lo, raw(la, lo), grading.ROAD_LIFT)

    def z_xy(xy):
        la, lo, _ = frame.enu_to_geodetic(xy[:, 0], xy[:, 1], 0.0)
        return np.asarray(elev(la, lo), np.float64)

    pad = radius / 111320.0
    padlon = pad / math.cos(math.radians(lat))
    keys = mapcells.cells_for_bbox(lat - pad, lon - padlon, lat + pad, lon + padlon)
    feats = {k: md.cell(k, layers=("roads",)).layers.get("roads") for k in keys}

    def in_r(xy):
        return np.hypot(xy[:, 0], xy[:, 1]) <= radius

    # --- ground roads: bumpiness and grades
    by_cls, seen = {}, set()
    ends = []           # (xy end (2,), z there, line, fid)
    lines = []
    for k, f in feats.items():
        if f is None or not len(f):
            continue
        specs = roadsmod.road_specs(f, None, None)
        for i in range(len(f)):
            sp = specs[i]
            fid = str(f.ids[i])
            if sp is None or fid in seen:
                continue
            seen.add(fid)
            for ring in roadsmod.ground_parts(f, i):
                ring = np.asarray(ring, np.float64)
                e, n, _ = frame.geodetic_to_enu(ring[:, 1], ring[:, 0], 0.0)
                xy, _s = grading.resample(np.stack([np.asarray(e), np.asarray(n)], 1), 10.0)
                if len(xy) < 2 or not in_r(xy).any():
                    continue
                z = z_xy(xy)
                lines.append((fid, sp, xy))
                ends += [(xy[0], z[0], fid), (xy[-1], z[-1], fid)]
                m = in_r(xy)
                if m.sum() >= 4:
                    zz = z[m]
                    d = by_cls.setdefault(sp.cls, {"d2": [], "grade": [], "km": 0.0})
                    d["d2"] += list(np.diff(zz, 2))
                    gr = np.abs(np.diff(zz)) / 10.0
                    if len(gr) and gr.max() > d.get("worst", (0.0,))[0]:
                        d["worst"] = (float(gr.max()), [round(float(v)) for v in xy[m][int(gr.argmax())]], fid[:8])
                    d["grade"] += list(gr)
                    d["km"] += 0.01 * (m.sum() - 1)
    roads_out = {c: {"km": round(d["km"], 1), "d2_rms": round(float(np.sqrt(np.mean(np.square(d["d2"])))), 3)
                     if d["d2"] else None, "grade_p95": round(_pct(d["grade"], 95), 3),
                     "grade_max": round(float(max(d["grade"])) if d["grade"] else float("nan"), 3),
                     "worst_at": d.get("worst", (0, None, None))[1:]}
                 for c, d in sorted(by_cls.items(), key=lambda kv: -kv[1]["km"])}

    # --- joins: an end of one road on another road, the height step to that road's line
    steps = []
    for p, z, fid in ends:
        if np.hypot(*p) > radius:
            continue
        for fid2, sp2, xy2 in lines:
            if fid2 == fid:
                continue
            a, b = xy2[:-1], xy2[1:]
            ab = b - a
            L2 = np.maximum((ab * ab).sum(1), 1e-12)
            t = np.clip(((p - a) * ab).sum(1) / L2, 0, 1)
            q = a + t[:, None] * ab
            dd = np.hypot(*(q - p).T)
            j = int(dd.argmin())
            if dd[j] <= sp2.width / 2:
                steps.append(abs(float(z_xy(q[j][None])[0]) - z))
                break
    joins = {"n": len(steps), "p95": round(_pct(steps, 95), 3), "max": round(float(max(steps)) if steps else 0, 3)}

    # --- decks over ground roads
    bp = roadsmod.BridgeProfiles(frame, lambda key: md.cell(key, layers=("roads",)).layers.get("roads"), elev)
    box = (-radius, -radius, radius, radius)
    clear, onto = [], []
    try:
        parts = bp.parts([f for f in feats.values() if f is not None], box)
    except Exception as e:
        parts = []
        print("decks:", type(e).__name__, e, file=sys.stderr)
    gl = [xy for _f, _sp, xy in lines]
    pends = np.array([np.asarray(xy, np.float64)[k, :2] for _f, _s, xy, _z, _l in parts for k in (0, -1)
                      if len(xy) >= 2]).reshape(-1, 2)
    for fid, spec, xy, z, _lon in parts:
        xy = np.asarray(xy, np.float64)[:, :2]
        z = np.asarray(z, np.float64)
        if len(xy) < 2 or not np.isfinite(z).all():
            continue
        s = np.r_[0.0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
        for _pi, sc, *_ in roadsmod._crossings([xy], gl):
            if sc < 6.0 or sc > s[-1] - 6.0:
                continue
            p = np.array([[np.interp(sc, s, xy[:, 0]), np.interp(sc, s, xy[:, 1])]])
            clear.append(float(np.interp(sc, s, z)) - float(z_xy(p)[0]))
        for at in (0, -1):          # onto the deck at an abutment: the ground 10 m before its end to 10 m in
            e = xy[at]
            if (np.hypot(*(pends - e).T) < 0.75).sum() > 1 or s[-1] < 25.0:
                continue            # a joint with another part (or too short to tell)
            d = xy[1] - xy[0] if at == 0 else xy[-2] - xy[-1]
            d = d / max(np.hypot(*d), 1e-9)
            gz = float(z_xy((e - 10.0 * d)[None])[0])
            dz = float(np.interp(10.0 if at == 0 else s[-1] - 10.0, s, z))
            onto.append(abs(dz - gz) / 20.0)
    decks = {"crossings": len(clear), "clear_p05": round(_pct(clear, 5), 2), "clear_min": round(min(clear), 2)
             if clear else None, "under_4_5m": int(sum(c < 4.5 for c in clear)),
             "onto_grade_p95": round(_pct(onto, 95), 3), "onto_grade_max": round(max(onto), 3) if onto else None}
    return {"dsm": dsmmode.is_dsm(lat, lon), "roads": roads_out, "joins": joins, "decks": decks}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--at", required=True, help="lat,lon")
    ap.add_argument("--radius", type=float, default=700.0)
    ap.add_argument("--cache-dir", default=os.path.join(os.path.expanduser("~"), ".cache", "beamng-earth"))
    ap.add_argument("--compare", action="store_true", help="BNE_DSM_MODE off and on, each in its own process")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    lat, lon = (float(v) for v in a.at.split(","))
    if a.compare:
        res = {}
        for m in ("off", "on"):
            out = subprocess.run([sys.executable, os.path.abspath(__file__), "--at", a.at, "--radius", str(a.radius),
                                  "--cache-dir", a.cache_dir, "--json"], env=dict(os.environ, BNE_DSM_MODE=m),
                                 capture_output=True, text=True, check=True)
            res[m] = json.loads(out.stdout.strip().splitlines()[-1])
        print(f"{'':28}{'off':>22}{'on':>22}")
        for sec in ("joins", "decks"):
            for k in res["off"][sec]:
                print(f"{sec + '.' + k:28}{str(res['off'][sec][k]):>22}{str(res['on'][sec][k]):>22}")
        for c in res["off"]["roads"]:
            for k in ("d2_rms", "grade_p95", "grade_max"):
                print(f"{'roads.' + c + '.' + k:28}{str(res['off']['roads'][c][k]):>22}"
                      f"{str(res['on']['roads'].get(c, {}).get(k)):>22}")
        return
    r = audit(lat, lon, a.radius, a.cache_dir)
    print(json.dumps(r) if a.json else json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
