import math
import unittest
from types import SimpleNamespace

import numpy as np

from environment import BoatEnv
from hull_collision import hull_collides
from main_safety_guard import MainSafetyGuard
from main_safety_kernels import (packed_hulls, preview_hull_collides,
                                 preview_lidar_distances, preview_follow_steering)
from perception import lidar_hits_np
from utils import pure_pursuit
from vessel_dynamics import VesselParameters


class MainSafetyGuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        length, width, gap = 84.0, 16.0, 11.0
        hull = [(length * 0.50, 0.0), (length * 0.12, width),
                (-length * 0.28, width * 0.85), (-length * 0.48, width * 0.6),
                (-length * 0.50, 0.0), (-length * 0.48, -width * 0.6),
                (-length * 0.28, -width * 0.85), (length * 0.12, -width)]
        cls.polygons = (
            [(x, y + gap) for x, y in hull],
            [(x, y - gap) for x, y in hull],
            [(length * 0.25, -gap * 0.85), (length * 0.25, gap * 0.85),
             (-length * 0.35, gap * 0.85), (-length * 0.35, -gap * 0.85)],
        )
        cls.packed, cls.lengths = packed_hulls(cls.polygons)

    def test_compiled_hull_matches_exact_reference(self):
        generator = np.random.default_rng(29)
        for _ in range(1000):
            x, y = generator.uniform(-30.0, 30.0, 2).astype(np.float32)
            heading = generator.uniform(-math.pi, math.pi)
            obstacles = np.column_stack((
                generator.uniform(-70.0, 70.0, (12, 2)),
                generator.uniform(8.0, 26.0, 12),
            )).astype(np.float32)
            reference = hull_collides((x, y), heading, obstacles, self.polygons)
            compiled = preview_hull_collides(float(x), float(y), heading,
                                             obstacles, self.packed, self.lengths)
            self.assertEqual(reference, compiled)

    def test_compiled_lidar_matches_reference(self):
        generator = np.random.default_rng(71)
        angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        for _ in range(100):
            position = generator.uniform(-100.0, 100.0, 2).astype(np.float32)
            heading = generator.uniform(-math.pi, math.pi)
            obstacles = np.column_stack((
                generator.uniform(-300.0, 300.0, (40, 2)),
                generator.uniform(8.0, 26.0, 40),
            )).astype(np.float32)
            reference, _, _ = lidar_hits_np(position, heading, angles,
                                            obstacles, 320.0)
            compiled = preview_lidar_distances(float(position[0]),
                                               float(position[1]), heading,
                                               obstacles, 320.0)
            self.assertLess(float(np.max(np.abs(reference - compiled))), 0.03)

    def test_shadow_steering_matches_gap_follower(self):
        generator = np.random.default_rng(42)
        angles = np.linspace(-np.pi, np.pi, 180, endpoint=False)
        params = {'steer_gain': 1.1, 'steer_alpha': 0.3515,
                  'avoid_normal': 0.05, 'avoid_em': 0.7,
                  'em_enter': 125.0, 'em_exit': 160.0,
                  'em_hold_frames': 18, 'pwm_rng': 270.36}
        for index in range(100):
            x, y = generator.uniform(100, 900, 2)
            heading = generator.uniform(-math.pi, math.pi)
            yaw = generator.uniform(-1.0, 1.0)
            previous = generator.uniform(-1.0, 1.0)
            distances = generator.uniform(15.0, 320.0, 180).astype(np.float32)
            path = (np.cumsum(generator.normal(0.0, 8.0, (90, 2)), axis=0)
                    + np.array([x, y]))
            has_waypoint = index % 2 == 0
            emergency = index % 3 == 0
            cooldown = index % 20
            shadow = SimpleNamespace(
                dt=0.04, steer_timer=0.0, boat_pos=np.array([x, y]),
                boat_heading=heading, boat_ang_vel=yaw,
                prev_steer=previous, params=params, lidar_beams=180,
                rel_angles=angles, emergency_mode=emergency,
                emergency_cooldown=cooldown,
                current_wp={'pos': np.array([200.0, 200.0])} if has_waypoint else None,
                pursuit_target=pure_pursuit(path, np.array([x, y]), lookahead=70),
                target=np.array([1740.0, 300.0]), closest_avoid_hit=None,
            )
            expected = BoatEnv.update_steering(shadow, distances)
            actual, next_steer, next_emergency, next_cooldown, _ = \
                preview_follow_steering(
                    x, y, heading, yaw, previous, emergency, cooldown,
                    path, distances, angles, params['steer_gain'],
                    params['steer_alpha'], params['avoid_normal'],
                    params['avoid_em'], params['em_enter'], params['em_exit'],
                    params['em_hold_frames'], has_waypoint,
                )
            self.assertAlmostEqual(float(expected), float(actual), places=6)
            self.assertAlmostEqual(shadow.prev_steer, next_steer, places=6)
            self.assertEqual(shadow.emergency_mode, next_emergency)
            self.assertEqual(shadow.emergency_cooldown, next_cooldown)

    def test_safe_command_and_shadow_leave_live_state_unchanged(self):
        position = np.array([100.0, 200.0], dtype=np.float32)
        velocity = np.array([20.0, 0.0])
        env = SimpleNamespace(
            dt=0.04, frame=3, manual_mode=False, linetrace_mode=False,
            boat_pos=position, boat_vel=velocity, boat_heading=0.0,
            boat_ang_vel=0.0, current_fwd=1000.0,
            dynamic_obstacles=np.empty((0, 3), dtype=np.float32),
            bezier_path=None, current_wp=None, target=np.array([1740.0, 300.0]),
            left_hull_local=self.polygons[0], right_hull_local=self.polygons[1],
            deck_local=self.polygons[2], rel_angles=np.linspace(-np.pi, np.pi, 180, endpoint=False),
            lidar_range=320, mass=10.0, drag=0.2, rot_drag=0.8, inertia=4.5,
            prev_steer=0.0, emergency_mode=False, emergency_cooldown=0,
            min_wide_dist=320.0,
            dynamics=VesselParameters(),
            physics_state=lambda: np.array([2.0, 4.0, 0.0, 0.4, 0.0,
                                            0.0, 0.0, 0.0]),
            pwm_to_thrust=lambda pwm: (pwm-1500)/400*25.0,
            params={'mom_coeff': 0.00665, 'steer_gain': 1.1,
                    'yaw_command_gain': 5.0,
                    'steer_alpha': 0.3515, 'avoid_normal': 0.05,
                    'avoid_em': 0.7, 'em_enter': 125.0, 'em_exit': 160.0,
                    'em_hold_frames': 18, 'pwm_rng': 270.36},
        )
        guard = MainSafetyGuard()
        self.assertEqual((1500, 1500), guard.command(env, 1500, 1500, 0.0))
        self.assertEqual(0, guard.predictions)
        guard._preview(env, 1500, 1500)
        np.testing.assert_array_equal(position, np.array([100.0, 200.0], dtype=np.float32))
        np.testing.assert_array_equal(velocity, np.array([20.0, 0.0]))
        self.assertEqual(0.0, env.boat_heading)
        self.assertEqual(1000.0, env.current_fwd)


if __name__ == '__main__':
    unittest.main()
