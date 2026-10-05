"""Numerical and ordering contracts for behavior-neutral planning reuse."""
import unittest
from unittest.mock import patch

import numpy as np

import passage_guidance
from fast_command_arrays import slew_yaw, slew_yaw_reference
from navigation_map import NavigationMap
from passage_geometry import physical_hull_polygons
from perception import lidar_hits_np


class PerformanceReuseTests(unittest.TestCase):
    def test_compiled_slew_is_bit_identical_to_numpy_reference(self):
        rng = np.random.default_rng(731)
        for samples, horizon in ((1, 1), (128, 40), (7, 19)):
            for previous in (0., -.0, .5, -.5, np.nan):
                with self.subTest(samples=samples, horizon=horizon, previous=previous):
                    reference = rng.normal(size=(samples, horizon, 2))
                    reference[0, 0, 1] = previous+.1
                    actual = reference.copy()
                    slew_yaw_reference(reference, previous, .1)
                    slew_yaw(actual, previous, .1)
                    np.testing.assert_array_equal(actual.view(np.uint64),
                                                  reference.view(np.uint64))
        for dtype in (np.float64, np.float32):
            reference = rng.normal(size=(7, 10, 2)).astype(dtype)
            previous = rng.normal(size=7)
            actual = reference.copy()
            slew_yaw_reference(reference, previous, .1)
            slew_yaw(actual, previous, .1)
            np.testing.assert_array_equal(actual, reference)

    def test_batch_norms_match_scalar_at_observation_thresholds(self):
        rng = np.random.default_rng(731)
        values = rng.uniform(-40., 40., (1000, 2))
        for threshold in (.12, .35, .5):
            edge = np.nextafter(threshold, np.inf)
            values = np.vstack((values, [threshold, 0.], [edge, 0.]))
        expected = np.array([np.linalg.norm(row) for row in values])
        np.testing.assert_array_equal(np.linalg.norm(values, axis=1), expected)

    def test_observed_tracks_match_scalar_norm_updates(self):
        rng = np.random.default_rng(42)
        obstacles = np.column_stack((rng.uniform(1., 35., 40),
                                     rng.uniform(1., 11., 40),
                                     rng.uniform(.1, .5, 40)))
        angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        scalar, batched = NavigationMap(36., 12.6), NavigationMap(36., 12.6)
        norm = np.linalg.norm

        def scalar_norm(values, axis=None):
            if axis == 1:
                return np.array([norm(row) for row in values])
            return norm(values, axis=axis)

        for tick in range(40):
            position = np.array([2.+tick*.4, 6.+np.sin(tick*.1)])
            ranges, _, _ = lidar_hits_np(position*50, tick*.04, angles,
                                        obstacles*50, 320, (0, 0, 1800, 630))
            args = (position, tick*.04, angles, ranges/50., 6.4, tick*.12)
            with patch.object(np.linalg, 'norm', side_effect=scalar_norm):
                scalar.observe(*args)
            batched.observe(*args)
            np.testing.assert_array_equal(batched.obstacles, scalar.obstacles)
            np.testing.assert_array_equal(batched.free_seen, scalar.free_seen)

    def test_single_combined_sort_preserves_family_order_and_ties(self):
        hull = physical_hull_polygons()
        position, goal = np.array([2., .9]), np.array([12., .9])
        obstacles = np.array([[5., 2.1, .3], [8., 2.1, .3],
                              [8., 6., .3], [5., 6., .3]])
        def families(sort):
            return (passage_guidance.observed_passages(
                obstacles, position, 0., goal, .2, hull_polygons=hull, sort=sort)+
                passage_guidance.observed_wall_passages(
                    obstacles, position, 0., goal, .2, 36., 12.6, hull, sort=sort))
        key = lambda p: norm(p.center-position)+.5*norm(goal-p.center)
        norm = np.linalg.norm
        old = sorted(families(True), key=key)
        new = sorted(families(False), key=key)
        self.assertTrue(old)
        self.assertEqual([(p.obstacle_pair, p.wall) for p in old],
                         [(p.obstacle_pair, p.wall) for p in new])
        for a, b in zip(old, new):
            np.testing.assert_array_equal(a.center, b.center)
            np.testing.assert_array_equal(a.tangent, b.tangent)
            self.assertEqual(a.current_required_width, b.current_required_width)


if __name__ == '__main__':
    unittest.main()
