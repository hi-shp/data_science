import unittest

import numpy as np

from dynamic_path_feasibility import (packed_parameters, shadow_allocate,
                                      shadow_integrate, shadow_move_obstacles,
                                      splice_path, retain_feasible_previous)
from vessel_dynamics import VesselParameters, allocate, integrate


class DynamicsPathPreviewTests(unittest.TestCase):
    def test_feasible_incumbent_survives_immaterial_new_route(self):
        self.assertTrue(retain_feasible_previous(200., 195., 35., 36., 12., 150.))
        self.assertFalse(retain_feasible_previous(200., 170., 35., 36., 12., 150.))
        self.assertFalse(retain_feasible_previous(200., 195., 35., 50., 12., 150.))

    def test_moving_obstacle_preview_matches_environment_equation(self):
        base = np.array([[627., 331., 17.], [574., 221., 17.]], dtype=np.float32)
        predicted = base.copy()
        for frame in (1, 100, 475):
            shadow_move_obstacles(base, predicted, frame, 0.04)
            phase = frame * 0.04 + base[:, 0] * 0.05 + base[:, 1] * 0.05
            expected_x = base[:, 0] + np.sin(phase) * base[:, 2] * 0.2
            expected_y = base[:, 1] + np.cos(phase * 1.2) * base[:, 2] * 0.2
            np.testing.assert_allclose(predicted[:, 0], expected_x, atol=2e-5)
            np.testing.assert_allclose(predicted[:, 1], expected_y, atol=2e-5)

    def test_splice_preserves_old_heading_at_join(self):
        old = np.array([[0., 0.], [20., 0.], [40., 0.], [60., 0.],
                        [80., 0.], [100., 0.]], dtype=np.float32)
        joined = splice_path(old, np.array([5., 0.]), np.array([140., 30.]),
                             np.empty((0, 3), dtype=np.float32), 25., 30., 20.)
        self.assertIsNotNone(joined)
        np.testing.assert_allclose(joined[0], [5., 0.], atol=1e-5)
        np.testing.assert_allclose(joined[-1], [140., 30.], atol=1e-5)
        old_tangent = joined[2] - joined[1]
        new_tangent = joined[3] - joined[2]
        self.assertLess(abs(np.arctan2(*new_tangent[::-1]) -
                            np.arctan2(*old_tangent[::-1])), 0.02)

    def test_shadow_allocator_and_integrator_match_frozen_dynamics(self):
        parameters = VesselParameters(
            mass_kg=20.0, yaw_inertia_kg_m2=3.8,
            surge_linear_drag=6.0, surge_quadratic_drag=7.116009950310329,
            yaw_linear_drag=8.0, yaw_quadratic_drag=5.0,
            yaw_response_s=0.65, yaw_rate_gain_Nm_s=11.76923076923077,
            cruise_speed_m_s=1.5)
        packed = packed_parameters(parameters)
        rng = np.random.default_rng(45)
        for _ in range(100):
            state = np.array([rng.uniform(0, 20), rng.uniform(0, 20),
                              rng.uniform(-3, 3), rng.uniform(-1, 2),
                              rng.uniform(-0.3, 0.3), rng.uniform(-0.5, 0.5),
                              rng.uniform(-25, 25), rng.uniform(-25, 25)])
            speed, yaw = rng.uniform(-0.5, 2), rng.uniform(-0.7, 0.7)
            expected_left, expected_right = allocate(state, speed, yaw,
                                                     parameters)
            left, right = shadow_allocate(state, speed, yaw, packed)
            self.assertAlmostEqual(left, expected_left, places=12)
            self.assertAlmostEqual(right, expected_right, places=12)
            expected = integrate(state, left, right, 0.04, parameters)
            actual = shadow_integrate(state, left, right, 0.04, packed)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
