import unittest

import mujoco
import numpy as np

from scripts.robocon_2027_full_map import build_full_map


class FullMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path, cls.manifest = build_full_map()
        cls.model = mujoco.MjModel.from_xml_path(str(cls.path))
        cls.names = [mujoco.mj_id2name(cls.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
                     for i in range(cls.model.ngeom)]

    def geom(self, name):
        return self.model.geom(name)

    def test_one_world_contains_all_eight_obstacles(self):
        self.assertEqual(len(self.manifest['stations']), 8)
        self.assertEqual(len([name for name in self.names if name.startswith('rc27_full_slalom_pole_')]), 6)
        self.assertEqual(len([name for name in self.names if name.startswith('rc27_full_tire_c')]), 128)
        self.assertEqual(len([name for name in self.names if name.startswith('rc27_full_plum_block_')]), 8)
        for name in ('low_frame_roof', 'beam', 'gully_takeoff_platform',
                     'gully_landing_platform', 'wall', 'plum_entry',
                     'pit_guard_-1', 'ground_before_gap', 'ground_after_gap'):
            self.assertIn('rc27_full_'+name, self.names)

    def test_rule_size_beam_and_wall(self):
        np.testing.assert_allclose(self.geom('rc27_full_beam').size[:3], [1.5, 0.2, 0.05])
        np.testing.assert_allclose(self.geom('rc27_full_wall').size[:3], [0.04, 0.4, 0.15])

    def test_gully_has_no_colliding_ground_under_gap(self):
        left, right = self.manifest['gap_interval_m']
        self.assertAlmostEqual(right-left, 0.36)
        middle = (left+right)/2
        for name in ('rc27_full_ground_before_gap', 'rc27_full_ground_after_gap'):
            geom = self.geom(name)
            self.assertFalse(abs(middle-geom.pos[0]) < geom.size[0])
        floor = self.geom('floor')
        self.assertEqual(int(floor.contype), 0)
        self.assertEqual(int(floor.conaffinity), 0)

    def test_slalom_finish_is_on_ground(self):
        zone = self.geom('rc27_full_slalom_zone_finish')
        ground = self.geom('rc27_full_ground_after_gap')
        self.assertLess(abs(zone.pos[1]), ground.size[1])


if __name__ == '__main__':
    unittest.main()
