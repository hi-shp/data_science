"""Wall marker suppression must never modify navigation sensor data."""
import unittest
import numpy as np
from lidar_hit_display import obstacle_hit_mask


class LidarHitDisplayTests(unittest.TestCase):
    def test_all_four_walls_hidden_buoy_hits_retained(self):
        x = np.array([0., 1840., 80., 80., 120., np.nan], dtype=np.float32)
        y = np.array([250., 250., 0., 644., 160., np.nan], dtype=np.float32)
        before_x, before_y = x.copy(), y.copy()
        np.testing.assert_array_equal(obstacle_hit_mask(x, y, 1840., 644.),
                                      [False, False, False, False, True, False])
        np.testing.assert_array_equal(x, before_x)
        np.testing.assert_array_equal(y, before_y)

    def test_boundary_roundoff_does_not_hide_interior_hits(self):
        edge = np.nextafter(np.float32(1840.), np.float32(0.))
        x = np.array([edge, 1839.99, 0.01], dtype=np.float32)
        y = np.full(3, 100., dtype=np.float32)
        np.testing.assert_array_equal(obstacle_hit_mask(x, y, 1840., 644.),
                                      [False, True, True])

    def test_scalar_legacy_3d_and_empty_hits(self):
        self.assertFalse(obstacle_hit_mask(100., 0., 1840., 644.))
        self.assertTrue(obstacle_hit_mask(100., 10., 1840., 644.))
        self.assertEqual(obstacle_hit_mask([], [], 1840., 644.).size, 0)


if __name__ == '__main__':
    unittest.main()
