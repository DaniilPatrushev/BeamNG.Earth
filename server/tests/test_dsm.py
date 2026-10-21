"""Surface-model ground (dsmmode.py; Moscow, 2026-10-21): road profiles smoothed harder and capped, roads meeting at
one height at junctions, every class's overpass approaches raised, no retaining walls; lidar places unchanged."""
import os
import sys
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import dsmmode  # noqa: E402
import grading  # noqa: E402
import roads  # noqa: E402
from test_roads import features, FRAME  # noqa: E402
from test_grading import FakeMD  # noqa: E402
from test_approaches import overpass, flat_elev  # noqa: E402


def dsm(on):
    return unittest.mock.patch.dict(os.environ, {"BNE_DSM_MODE": "on" if on else "off"})


def at(xs, ys):
    la, lo, _ = FRAME.enu_to_geodetic(np.asarray(xs, np.float64), np.asarray(ys, np.float64), 0.0)
    return la, lo


class ModeTest(unittest.TestCase):
    def test_modes(self):
        with dsm(True):
            self.assertTrue(dsmmode.is_dsm(55.75, 37.62))
        with dsm(False):
            self.assertFalse(dsmmode.is_dsm(55.75, 37.62))
        with unittest.mock.patch.dict(os.environ, {"BNE_DSM_MODE": "auto"}):
            dsmmode.register(None)
            self.assertFalse(dsmmode.is_dsm(55.75, 37.62), "no Dem registered: the lidar settings")

    def test_auto_asks_the_dem_once_per_tile(self):
        class Dem:
            calls = 0

            def _download(self, src, z, x, y, kind="demand"):
                Dem.calls += 1
                return None             # no z13 tile: no lidar
        with unittest.mock.patch.dict(os.environ, {"BNE_DSM_MODE": "auto"}):
            dsmmode.register(Dem())
            try:
                self.assertTrue(dsmmode.is_dsm(55.75, 37.62))
                n = Dem.calls
                self.assertTrue(dsmmode.is_dsm(55.7501, 37.6201))
                self.assertEqual(Dem.calls, n, "cached per z12 tile")
            finally:
                dsmmode.register(None)


class ProfileTest(unittest.TestCase):
    def test_bumps_gone_grade_capped(self):
        s = np.arange(0, 1000, grading.PROFILE_STEP)
        ground = 150.0 + 0.02 * s
        rng = np.random.default_rng(1)
        # a surface model: 1-2 m noise, tree / roof bumps standing up 3-8 m over 30-90 m
        bumps = np.zeros(len(s))
        for c in rng.uniform(50, 950, 12):
            bumps += rng.uniform(3, 8) * np.exp(-0.5 * ((s - c) / rng.uniform(15, 45)) ** 2)
        z = ground + rng.normal(0, 1.0, len(s)) + bumps
        spec = roads.Spec(7.0, "paved", False, "primary")
        old = grading.smooth_profile(z, grading.PROFILE_STEP, grading.SIGMA_M)
        new = grading.dsm_profile(z, grading.PROFILE_STEP, spec)
        d2 = lambda p: np.sqrt(np.mean(np.diff(p[::5], 2) ** 2))        # 2nd difference per 10 m
        self.assertLess(d2(new), d2(old) / 3, (d2(new), d2(old)))
        self.assertLessEqual(np.abs(np.diff(new)).max() / grading.PROFILE_STEP, grading.DSM_GRADE_MAJOR + 1e-9)
        self.assertLess(np.median(new - ground), np.median(old - ground) - 1.0, "nearer the ground than the canopy")

    def test_grade_limit(self):
        z = np.r_[np.zeros(20), 4.0, np.zeros(20)]
        c = grading.grade_limit(z, 2.0, 0.1)
        self.assertLessEqual(np.abs(np.diff(c)).max(), 0.2 + 1e-9)
        self.assertAlmostEqual(c[20], 2.0 + 0.0, delta=0.11, msg="half way: neither cut to the floor nor kept")
        ramp = 0.05 * np.arange(50.0)
        self.assertTrue(np.allclose(grading.grade_limit(ramp, 1.0, 0.1), ramp), "a gentle grade unchanged")


class SnapTest(unittest.TestCase):
    def plan(self, pts, z, cls="residential"):
        xy = np.asarray(pts, np.float64)
        xy, _ = grading.resample(xy, 4.0)
        zz = np.full(len(xy), float(z)) if np.isscalar(z) else z(xy)
        return grading.Plan(xy, zz, 4.0, True), roads.Spec(7.0, "paved", False, cls)

    def test_t_junction_takes_the_higher_class(self):
        main, ms = self.plan([(-300, 0), (300, 0)], 10.0, "primary")
        side, ss = self.plan([(0, 3), (0, 300)], 12.0)
        z = grading.snap_profile(side, ss, (True, True), [(main, ms, (True, True))])
        self.assertAlmostEqual(float(z[0]), 10.0, places=6)
        self.assertAlmostEqual(float(z[-1]), 12.0, places=6, msg="faded out along the road")
        self.assertIsNone(grading.snap_profile(main, ms, (True, True), [(side, ss, (True, True))]),
                          "the primary keeps its own")

    def test_ways_of_one_class_meet_at_the_mean(self):
        a, sa = self.plan([(-300, 0), (0, 0)], 10.0)
        b, sb = self.plan([(0, 0), (300, 0)], 11.0)
        za = grading.snap_profile(a, sa, (True, True), [(b, sb, (True, True))])
        zb = grading.snap_profile(b, sb, (True, True), [(a, sa, (True, True))])
        self.assertAlmostEqual(float(za[-1]), 10.5, places=6)
        self.assertAlmostEqual(float(zb[0]), 10.5, places=6)

    def test_crossing_at_grade(self):
        a, sa = self.plan([(-300, 0), (300, 0)], 10.0)
        b, sb = self.plan([(0, -300), (0, 300)], 12.0)
        za = grading.snap_profile(a, sa, (True, True), [(b, sb, (True, True))])
        zb = grading.snap_profile(b, sb, (True, True), [(a, sa, (True, True))])
        ia, ib = np.abs(a.xy[:, 0]).argmin(), np.abs(b.xy[:, 1]).argmin()
        self.assertAlmostEqual(float(za[ia]), float(zb[ib]), places=6)
        self.assertAlmostEqual(float(za[0]), 10.0, places=6)

    def test_cut_ends_and_big_steps_ignored(self):
        main, ms = self.plan([(-300, 0), (300, 0)], 10.0, "primary")
        side, ss = self.plan([(0, 3), (0, 300)], 12.0)
        self.assertIsNone(grading.snap_profile(side, ss, (False, True), [(main, ms, (True, True))]))
        high, hs = self.plan([(0, 3), (0, 300)], 30.0)
        self.assertIsNone(grading.snap_profile(high, hs, (True, True), [(main, ms, (True, True))]))


class GraderDsmTest(unittest.TestCase):
    def test_t_junction_continuous_in_the_grader(self):
        f = features([[(-300, 0), (300, 0)], [(0, 0), (0, 300)]], ["primary", "residential"], ids=["m", "s"])

        def elev(la, lo):           # the side road's ground 2 m higher than the main road's
            e, n, _ = FRAME.geodetic_to_enu(la, lo, 0.0)
            return 10.0 + 2.0 * (np.asarray(n) > 6.0)
        with dsm(True):
            g = grading.Grader(FRAME, FakeMD(f), elev)
            la, lo = at([0.0, 0.0, 0.0], [0.0, 8.0, 14.0])
            z = g.apply(la, lo, elev(la, lo))
        self.assertLess(abs(z[1] - z[0]) / 8.0, 0.06, z)      # a gentle ramp, no step
        self.assertLess(abs(z[2] - z[1]) / 6.0, 0.06, z)

    def test_motorway_overpass_approaches_raised(self):
        # a motorway bridge over a local road on flat ground: on lidar its fills are in the DEM and it is left; on a
        # surface model they are not, and the deck lay on the road under it (MKAD, 2026-10-21)
        a, b = (400 - 35) / 800, (400 + 35) / 800
        f = features([[(-400, 0), (400, 0)], [(-12, -500), (-12, 500)]], ["motorway", "residential"],
                     flags=[(0, "is_bridge", a, b)], ids=["mkad", "local"])
        res = {}
        for on in (False, True):
            with dsm(on):
                g = grading.Grader(FRAME, FakeMD(f), flat_elev)

                def elev(la, lo):
                    return g.apply(la, lo, flat_elev(la, lo), grading.ROAD_LIFT)
                bp = roads.BridgeProfiles(FRAME, lambda key: f, elev)
                (_fid, _spec, xy, z, _lon), = bp.parts([f], (-100, -100, 100, 100))
                xy = np.asarray(xy)
                lane = float(elev(*at([-12.0], [0.0]))[0])
                res[on] = float(np.interp(-12.0, xy[:, 0], z)) - lane
        self.assertLess(res[False], 4.0)
        self.assertGreaterEqual(res[True], roads.DECK_ROAD_CLEAR - 0.1, res)

    def test_divided_road_both_ramps_raised(self):
        # a divided trunk's two carriageways 14 m apart, each on its own bridge over a local road: each ramp's fill
        # capped the other's to nothing, the decks lay 1.6 m over the road (Leningradskoye shosse, 2026-10-21)
        a, b = (400 - 25) / 800, (400 + 25) / 800
        f = features([[(-400, 0), (400, 0)], [(400, 14), (-400, 14)], [(-5, -500), (-5, 500)]],
                     ["trunk", "trunk", "secondary"],
                     flags=[(0, "is_bridge", a, b), (1, "is_bridge", a, b)], ids=["east", "west", "local"])
        with dsm(True):
            g = grading.Grader(FRAME, FakeMD(f), flat_elev)

            def elev(la, lo):
                return g.apply(la, lo, flat_elev(la, lo), grading.ROAD_LIFT)
            bp = roads.BridgeProfiles(FRAME, lambda key: f, elev)
            parts = bp.parts([f], (-100, -100, 100, 100))
            lane = float(elev(*at([-5.0], [7.0]))[0])
            for _fid, _spec, xy, z, _lon in parts:
                xy = np.asarray(xy)
                self.assertGreaterEqual(float(np.interp(-5.0, xy[:, 0], z)) - lane, roads.DECK_ROAD_CLEAR - 0.6)

    def test_lidar_unchanged(self):
        f = overpass()
        with dsm(False):
            g = grading.Grader(FRAME, FakeMD(f), flat_elev)
            la, lo = at(np.linspace(-200, 200, 41), np.zeros(41))
            a = g.apply(la, lo, flat_elev(la, lo))
        with unittest.mock.patch.dict(os.environ, {"BNE_DSM_MODE": "auto"}):
            dsmmode.register(None)
            g = grading.Grader(FRAME, FakeMD(f), flat_elev)
            b = g.apply(la, lo, flat_elev(la, lo))
        self.assertTrue(np.array_equal(a, b))

    def test_no_walls_on_a_surface_model(self):
        f = features([[(-200, 0), (200, 0)]], ["residential"])

        def elev(la, lo):           # the road in a 4 m cut
            e, n, _ = FRAME.geodetic_to_enu(la, lo, 0.0)
            return 10.0 + 4.0 * (np.abs(np.asarray(n)) > 9.0)
        flat = lambda la, lo: np.full(np.shape(la), 10.0)
        outs = {}
        for on in (False, True):
            with dsm(on):
                g = grading.Grader(FRAME, FakeMD(f), flat)
                la, lo = at(np.zeros(2), [0.0, 11.0])
                outs[on] = g.apply(la, lo, elev(la, lo))
        self.assertLess(outs[True][1], outs[False][1] - 0.2, "surface model: a gentler slope, no wall's step")


class ObjectsTest(unittest.TestCase):
    def test_ml_box_on_a_wide_road(self):
        import buildings
        road = [(np.array([[-100.0, 0.0], [100.0, 0.0]]), 7.0)]          # a 14 m carriageway
        box = lambda cx, cy, r=2.5: np.array([[cx - r, cy - r], [cx + r, cy - r], [cx + r, cy + r], [cx - r, cy + r],
                                              [cx - r, cy - r]])
        self.assertTrue(buildings._on_carriageway(box(0.0, 6.0), road), "a lorry in the outer lane")
        self.assertFalse(buildings._on_carriageway(box(0.0, 14.0), road), "a kiosk beside the road")
        self.assertTrue(buildings._on_carriageway(box(0.0, 9.5, 3.0), road), "straddling the edge")

    def test_props_on_the_ground_where_they_stand(self):
        import furniture
        rows = [["signal", 10.0, 5.0, 20.15, 0.0, 1.0, 9.4, 0.0]]
        furniture._on_ground(rows, lambda q: np.full(len(q), 17.0), 0.15)
        self.assertAlmostEqual(rows[0][3], 17.15)
        rows = [["signal", 10.0, 5.0, 20.15, 0.0, 1.0, 9.4, 0.0]]
        furniture._on_ground(rows, None, 0.15)
        self.assertAlmostEqual(rows[0][3], 20.15, msg="no ground: as before")


class LocalDemTest(unittest.TestCase):
    def test_local_cells_ahead_of_the_rest(self):
        import tempfile
        import dem as demmod
        import localdem
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "local", "fabdem"))
            # one cell, 55..56 N 37..38 E: a plane rising 100 m per degree east, 300 x 200 px
            lon = 37 + (np.arange(200) + 0.5) / 200
            np.save(os.path.join(d, "local", "fabdem", "N55E037.npy"),
                    np.tile((150.0 + 100.0 * (lon - 37)).astype(np.float32), (300, 1)))
            ld = localdem.LocalDem(os.path.join(d, "local"))
            self.assertTrue(ld)
            h = ld.sample(np.array([55.5, 55.5, 54.5]), np.array([37.25, 37.75, 37.5]))
            self.assertAlmostEqual(h[0], 175.0, delta=0.01)
            self.assertAlmostEqual(h[1], 225.0, delta=0.01)
            self.assertTrue(np.isnan(h[2]))

            class Fetch:            # no network: the base tiles must not be asked for a tile all inside the cell
                def register_chain(self, *a):
                    pass
            dm = demmod.Dem(os.path.join(d), fetcher=Fetch())
            n = 2 ** 15
            x = int((37.5 + 180) / 360 * n)
            y = int((1 - np.arcsinh(np.tan(np.radians(55.5))) / np.pi) / 2 * n)
            t = dm._load_tile((15, x, y))
            self.assertTrue(np.isfinite(t).all())
            self.assertAlmostEqual(float(t.mean()), 200.0, delta=1.0)
            self.assertEqual(dm.served[(15, x, y)], "local:fabdem")


if __name__ == "__main__":
    unittest.main()
