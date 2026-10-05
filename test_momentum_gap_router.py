import math
import unittest
from types import SimpleNamespace

import numpy as np

from momentum_gap_router import (MomentumGapRouter, infer_buoy_origins,
                                 perceived_circles, portal_crossing,
                                 portal_interval, portal_wall_interval,
                                 select_near_tie, _simulate,
                                 _portal_route_order, wall_lidar_hits,
                                 trajectory_bezier_reference)
from main_safety_kernels import (packed_hulls, preview_hull_wall_clearance,
                                 preview_hull_wall_turning_room)
from dynamic_path_feasibility import packed_parameters
from dynamic_path_feasibility import shadow_allocate, shadow_integrate
from vessel_dynamics import VesselParameters
from ui_renderer import momentum_display_path, momentum_display_portal


class MomentumGapRouterTests(unittest.TestCase):
    def test_observation_fit_uses_current_lidar_arc(self):
        center = np.array([120.0, 20.0])
        angles = np.linspace(2.5, 3.7, 11)
        hits = center + 17.0 * np.column_stack((np.cos(angles), np.sin(angles)))
        fitted = perceived_circles(hits[:, 0], hits[:, 1],
                                   [center - np.array([12.0, 0.0])],
                                   np.array([0.0, 20.0]), 17.0)
        self.assertEqual(fitted.shape, (1, 3))
        self.assertLess(np.linalg.norm(fitted[0, :2] - center), 2.0)
        self.assertGreaterEqual(fitted[0, 2], 19.0)

    def test_moving_buoy_origin_reconstructs_current_observation(self):
        observed = np.array([[120.0, 50.0, 19.0]], dtype=np.float32)
        origins = infer_buoy_origins(observed, frame=60, dt=0.04)
        phase = 60 * 0.04 + 0.05 * (origins[0, 0] + origins[0, 1])
        self.assertAlmostEqual(origins[0, 0] + math.sin(phase) * 19.0 * 0.2,
                               observed[0, 0], delta=0.02)
        self.assertAlmostEqual(origins[0, 1] + math.cos(phase * 1.2) * 19.0 * 0.2,
                               observed[0, 1], delta=0.02)

    def test_safe_portal_interval_allows_off_midpoint_crossing(self):
        gap = {'c1': np.array([100.0, 0.0]),
               'c2': np.array([100.0, 150.0])}
        interval = portal_interval(gap, obstacle_radius=17.0,
                                   hull_half_width=25.0, margin_px=10.0)
        self.assertIsNotNone(interval)
        crossed, s = portal_crossing([90.0, 60.0], [110.0, 60.0],
                                     gap, interval, [200.0, 60.0])
        self.assertTrue(crossed)
        self.assertAlmostEqual(s, 0.4)
        crossed, _ = portal_crossing([90.0, 25.0], [110.0, 25.0],
                                     gap, interval, [200.0, 25.0])
        self.assertFalse(crossed)

    def test_safe_portal_segment_is_clipped_by_known_walls(self):
        gap = {'c1': np.array([100., 0.]),
               'c2': np.array([100., 200.])}
        clipped = portal_wall_interval(
            gap, (0., 1.), 0., [[(0., -10.), (0., 10.)]],
            10., 200., 150.)
        self.assertAlmostEqual(clipped[0], .10)
        self.assertAlmostEqual(clipped[1], .65)

    def test_near_tie_keeps_incumbent_but_unsafe_is_not_present(self):
        candidates = [(100.0, 0.0, 1.0, 'straight'),
                      (103.0, 1.0, 0.0, 'left_gentle')]
        self.assertEqual(select_near_tie(candidates, 'left_gentle')[3],
                         'left_gentle')
        self.assertEqual(select_near_tie(candidates, None)[3], 'straight')

    def test_episode_reset_clears_route_and_command_memory(self):
        router = MomentumGapRouter()
        router.previous_family = 'left_strong'
        router.last_result = {'family': 'left_strong'}
        router.last_frame = 500
        router.last_position = np.array([100.0, 20.0])
        router.last_scores = [('left_strong', 20.0, True)]
        router.no_progress_pair = (1, 2)
        router.no_progress_start_frame = 200
        router.reset_episode()
        self.assertIsNone(router.previous_family)
        self.assertIsNone(router.last_result)
        self.assertIsNone(router.last_position)
        self.assertEqual(router.last_frame, -1)
        self.assertEqual(router.last_scores, [])
        self.assertIsNone(router.no_progress_pair)
        self.assertIsNone(router.no_progress_start_frame)

    def test_selected_rollout_display_starts_at_current_position(self):
        rollout = np.array([[0., 0., 0.], [10., 0., 0.],
                            [20., 0., 0.], [30., 10., 0.]])
        display = momentum_display_path(rollout, [12., 1.])
        np.testing.assert_allclose(display[0], [12., 1.])
        np.testing.assert_allclose(display[1], [12., 0.])
        np.testing.assert_allclose(display[-1], [30., 10.])
        np.testing.assert_array_equal(rollout[0], [0., 0., 0.])

    def test_portal_display_uses_safe_segment_and_actual_crossing(self):
        gap = {'c1': np.array([100., 0.]), 'c2': np.array([100., 150.])}
        router = MomentumGapRouter()
        router.last_result = {'crossing_s': 0.4, 'portal': {**gap, 'interval': (52./150., 98./150.)}}
        env = SimpleNamespace(momentum_gap_router=router, current_wp=gap,
                              left_hull_local=[(0., 25.)],
                              right_hull_local=[(0., -25.)],
                              deck_local=[(0., 0.)], obs_r=17.,
                              target=np.array([200., 75.]),
                              map_w=300., sim_h=200.,
                              dynamics=SimpleNamespace(pixels_per_m=50.))
        first, second, safe_first, safe_second, crossing = momentum_display_portal(env)
        self.assertLess(safe_first[1], crossing[1])
        self.assertLess(crossing[1], safe_second[1])
        self.assertAlmostEqual(crossing[1], 60.)

    def test_full_hull_wall_gate_checks_front_side_rear_and_sweep(self):
        hulls = ([(42., -7.), (42., 7.), (-42., 7.), (-42., -7.)],
                 [(42., -27.), (42., -13.), (-42., -13.), (-42., -27.)],
                 [(18., -8.), (18., 8.), (-18., 8.), (-18., -8.)])
        polygons, lengths = packed_hulls(hulls)
        self.assertLess(preview_hull_wall_clearance(40., 300., 0.,
                            polygons, lengths, 1840., 620.), 0.)
        self.assertLess(preview_hull_wall_clearance(1800., 300., 0.,
                            polygons, lengths, 1840., 620.), 0.)
        self.assertLess(preview_hull_wall_clearance(500., 25., 0.,
                            polygons, lengths, 1840., 620.), 0.)
        self.assertLess(preview_hull_wall_clearance(500., 615., 0.,
                            polygons, lengths, 1840., 620.), 0.)
        self.assertGreater(preview_hull_wall_clearance(500., 300., 0.,
                            polygons, lengths, 1840., 620.), 10.)
        state = np.zeros(8, dtype=np.float64)
        state[0] = 40. / 50.
        state[1] = 300. / 50.
        safe = _simulate(state, np.zeros((2, 2)), np.empty((0, 3)),
                         polygons, lengths, packed_parameters(VesselParameters()),
                         .04, 2, 10., 0, 1840., 620.)
        self.assertFalse(safe[0])

    def test_terminal_wall_turning_room_uses_velocity_and_yaw(self):
        hulls = ([(42., -7.), (42., 7.), (-42., 7.), (-42., -7.)],
                 [(42., -27.), (42., -13.), (-42., -13.), (-42., -27.)],
                 [(18., -8.), (18., 8.), (-18., 8.), (-18., -8.)])
        polygons, lengths = packed_hulls(hulls)
        toward = preview_hull_wall_turning_room(
            500., 80., -math.pi/2, 1.2, 0., 0., polygons, lengths,
            1840., 620., 50., 10., .9)
        away = preview_hull_wall_turning_room(
            500., 80., math.pi/2, 1.2, 0., 0., polygons, lengths,
            1840., 620., 50., 10., .9)
        self.assertLess(toward, 0.)
        self.assertGreater(away, 0.)

    def test_geometric_portal_selection_has_no_midpoint_target(self):
        env = SimpleNamespace(
            boat_pos=np.array([20., 80.]), target=np.array([300., 120.]),
            clusters=[np.array([100., 20.]), np.array([100., 200.]),
                      np.array([160., 100.]), np.array([280., 100.])],
            cluster_ids=[0, 1, 2, 3], visited=set(), obs_r=17.,
            lidar_range=320., map_w=400., sim_h=300.,
            left_hull_local=[(-20., -25.), (20., -25.)],
            right_hull_local=[(-20., 25.), (20., 25.)],
            deck_local=[(-20., -5.), (20., 5.)],
            dynamics=SimpleNamespace(pixels_per_m=50.))
        options = MomentumGapRouter().portal_options(env)
        vertical = next(option for option in options
                        if set(option['pair']) == {0, 1})
        self.assertGreater(abs(float(vertical['pos'][1])-110.), 5.)
        self.assertFalse(any(set(option['pair']) == {2, 3}
                             for option in options))

    def test_compiled_portal_order_matches_reference_ternary(self):
        rng = np.random.default_rng(1234)
        for _ in range(40):
            first, axis, position, goal = rng.uniform(-100., 300., (4, 2))
            low, high = .2, .8
            left, right = low, high
            for _ in range(16):
                one = (2*left+right)/3
                two = (left+2*right)/3
                p_one, p_two = first+one*axis, first+two*axis
                d_one = np.linalg.norm(p_one-position)+np.linalg.norm(goal-p_one)
                d_two = np.linalg.norm(p_two-position)+np.linalg.norm(goal-p_two)
                if d_one <= d_two:
                    right = two
                else:
                    left = one
            reference = first+0.5*(left+right)*axis
            actual = _portal_route_order(
                first[0], first[1], axis[0], axis[1], position[0],
                position[1], goal[0], goal[1], low, high)
            np.testing.assert_allclose(actual[:2], reference, rtol=0, atol=1e-10)

    def test_wall_hits_cover_all_sides_without_fabricating_buoy_hits(self):
        env = SimpleNamespace(boat_pos=np.array([50., 50.]), boat_heading=0.,
            rel_angles=np.array([0., math.pi/2, math.pi, -math.pi/2]),
            lidar_range=320., map_w=100., sim_h=100.)
        empty = np.full(4, np.nan)
        distance, hx, hy = wall_lidar_hits(env, np.full(4, 320.), empty, empty)
        np.testing.assert_allclose(distance, 50.)
        np.testing.assert_allclose(hx, [100., 50., 0., 50.], atol=1e-5)
        np.testing.assert_allclose(hy, [50., 100., 50., 0.], atol=1e-5)
        self.assertTrue(np.isnan(empty).all())

    def test_wall_portal_uses_zero_wall_radius_and_real_buoy_radius(self):
        gap = {'c1': np.array([100., 0.]), 'c2': np.array([100., 160.]),
               'endpoint_radii': (0., 17.)}
        low, high = portal_interval(gap, 17., 27., 10.)
        self.assertAlmostEqual(low, 37./160.)
        self.assertAlmostEqual(high, 1.-54./160.)
        self.assertNotEqual(low, 1.-high)

    def test_bezier_reference_preserves_prediction_and_endpoints(self):
        prediction = np.column_stack((np.arange(9.), np.arange(9.)**2, np.zeros(9)))
        original = prediction.copy()
        reference = trajectory_bezier_reference(prediction)
        np.testing.assert_array_equal(prediction, original)
        np.testing.assert_allclose(reference[[0,-1]], prediction[[0,-1],:2])

    def test_safe_forward_returns_from_previous_reverse(self):
        hulls = ([(-42., -27.), (42., -27.), (42., -13.), (-42., -13.)],
                 [(-42., 13.), (42., 13.), (42., 27.), (-42., 27.)],
                 [(-18., -8.), (18., -8.), (18., 8.), (-18., 8.)])
        state = np.array([10., 6., 0., .2, 0., 0., 0., 0.])
        env = SimpleNamespace(boat_pos=np.array([500., 300.]), target=np.array([1500.,300.]),
            boat_heading=0., frame=1, dt=.04, map_w=1840., sim_h=620.,
            dynamics=VesselParameters(), left_hull_local=hulls[0],
            right_hull_local=hulls[1], deck_local=hulls[2],
            command_speed=-.35, command_yaw_rate=.2, physics_state=lambda: state)
        result = MomentumGapRouter().choose(env, None, None, None,
                                            observed=np.empty((0,3), dtype=np.float32))
        self.assertGreaterEqual(result['speed_command'], 0.)
        self.assertFalse(result['recovery'])
        self.assertTrue(result['safe'])
        self.assertLessEqual(abs(result['yaw_rate_command']-.2), .1*.04/.12+1e-12)
        params = packed_parameters(env.dynamics)
        left, right = shadow_allocate(state, result['speed_command'],
                                      result['yaw_rate_command'], params)
        expected = shadow_integrate(state, left, right, env.dt, params)
        np.testing.assert_allclose(result['prediction'][1],
            [expected[0]*50., expected[1]*50., expected[2]], atol=1e-9)

    def test_single_observed_buoy_can_generate_wall_portals(self):
        env = SimpleNamespace(boat_pos=np.array([60.,50.]), target=np.array([350.,50.]),
            clusters=[np.array([200.,100.])], cluster_ids=[1], visited=set(),
            obs_r=17., lidar_range=320., map_w=400., sim_h=300.,
            left_hull_local=[(-20.,-25.),(20.,-25.)],
            right_hull_local=[(-20.,25.),(20.,25.)],
            deck_local=[(-20.,-5.),(20.,5.)],
            dynamics=SimpleNamespace(pixels_per_m=50.))
        options = MomentumGapRouter().portal_options(env)
        self.assertTrue(any(gap['wall_id'] is not None for gap in options))
        self.assertTrue(all(gap['endpoint_radii'][0] == 0. for gap in options))


if __name__ == '__main__':
    unittest.main()
