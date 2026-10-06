import math
import unittest
from types import SimpleNamespace

import numpy as np

from portal_navigation import (choose_portal_crossing,
                               path_has_hull_clearance, remaining_path,
                               portal_crossing_status)


class PortalCrossingTests(unittest.TestCase):
    def setUp(self):
        self.dynamics = SimpleNamespace(pixels_per_m=50.0,
                                        cruise_speed_m_s=1.5,
                                        actuator_tau_s=0.25,
                                        max_yaw_rate_rad_s=1.2)
        # Elongated hull, 84 px long and 54 px wide, matching the projection
        # property of the frozen twin-hull collision geometry.
        self.hull = [[(-42, -27), (-42, 27), (42, 27), (42, -27)]]

    def crossing(self, width, position=(30, 10), downstream=(210, 10),
                 heading=0.0, previous_s=None):
        left = np.array((100.0, 0.0))
        right = np.array((100.0, float(width)))
        gap = {'c1': left, 'c2': right, 'pos': (left + right) / 2,
               'pair': (1, 2)}
        obs = np.array([[*left, 10.0], [*right, 10.0]])
        return choose_portal_crossing(gap, np.array(position, dtype=float),
                                      heading, np.array((50.0, 0.0)), 0.0,
                                      np.array(downstream, dtype=float), obs,
                                      self.hull, self.dynamics,
                                      previous_s=previous_s, plan_dt=0.12)

    def test_wide_portal_selects_off_midpoint_when_route_is_off_center(self):
        result = self.crossing(150)
        self.assertIsNotNone(result)
        self.assertLess(result['portal_s'], 0.5)
        self.assertGreaterEqual(result['portal_s'], result['portal_safe_interval'][0])

    def test_too_narrow_portal_is_not_a_waypoint(self):
        self.assertIsNone(self.crossing(70))

    def test_current_angle_does_not_reject_alignable_portal(self):
        result = self.crossing(110, heading=math.pi / 2)
        self.assertIsNotNone(result)
        self.assertLess(abs(result['portal_heading']), 0.4)

    def test_previous_crossing_changes_gradually_and_stays_safe(self):
        first = self.crossing(150, previous_s=0.65)
        self.assertIsNotNone(first)
        self.assertLess(abs(first['portal_s'] - 0.65), 0.1)
        low, high = first['portal_safe_interval']
        self.assertLessEqual(low, first['portal_s'])
        self.assertLessEqual(first['portal_s'], high)

    def test_path_rejects_hull_contact_even_when_centerline_is_clear(self):
        path = np.array([[0.0, 0.0], [100.0, 0.0]])
        obstacle = np.array([[50.0, 37.0, 5.0]])
        self.assertFalse(path_has_hull_clearance(path, obstacle, self.hull))
        obstacle[0, 1] = 100.0
        self.assertTrue(path_has_hull_clearance(path, obstacle, self.hull))

    def test_remaining_path_starts_at_segment_projection(self):
        path = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0]])
        remaining = remaining_path(path, np.array([10.0, 5.0]))
        self.assertTrue(np.allclose(remaining[0], [10.0, 5.0]))
        self.assertTrue(np.allclose(remaining[-1], [10.0, 10.0]))

    def test_portal_is_visited_only_after_safe_line_crossing(self):
        gap = self.crossing(150)
        crossed, before = portal_crossing_status(gap, np.array([95.0, 75.0]))
        self.assertFalse(crossed)
        crossed, after = portal_crossing_status(gap, np.array([105.0, 75.0]), before)
        self.assertTrue(crossed)
        self.assertLess(before, 0.0)
        self.assertGreater(after, 0.0)
        crossed, _ = portal_crossing_status(gap, np.array([105.0, -5.0]), before)
        self.assertFalse(crossed)


if __name__ == '__main__':
    unittest.main()
