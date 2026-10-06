"""Display interpolation stays on the unchanged path and out of control."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from pursuit_marker_display import PursuitDisplayMarker


class PursuitMarkerTests(unittest.TestCase):
    def marker(self, points, target, speed=0.):
        marker = PursuitDisplayMarker()
        marker.set_path(np.asarray(points, dtype=float), np.asarray(target),
                        1, speed, .12, 50.)
        marker.update(0., speed, .04)
        return marker

    def test_corner_following_does_not_cut_across_bezier(self):
        path = np.array([[0., 0.], [100., 0.], [100., 100.]])
        marker = self.marker(path, [90., 0.])
        marker.set_path(path, np.array([100., 20.]), 4, 0., .12, 50.)
        points = [marker.update(i/120., 0., .04) for i in range(1, 121)]
        self.assertTrue(any(point[1] > 0. for point in points))
        for x, y in points:
            self.assertTrue(y == 0. or x == 100., (x, y))

    def test_stale_planning_target_moves_on_render_frames_without_backtracking(self):
        path = np.array([[0., 0.], [500., 0.]])
        marker = self.marker(path, [70., 0.], speed=80.)
        positions = []
        for i in range(1, 61):
            if i % 6 == 0:
                # Same publication cadence as 1x: three .04-s physics steps
                # per six 120-Hz render frames. Quantized targets stay unchanged.
                target = np.array([70.+(i//6)*10., 0.])
                marker.set_path(path, target, 1+(i//6)*3, 80., .12, 50.)
            positions.append(marker.update(i/120., 80., .04)[0])
        delta = np.diff(positions)
        self.assertTrue(np.all(delta >= 0.))
        self.assertGreater(np.count_nonzero(delta > 0.), 50)
        self.assertLess(delta.max(), 4.)  # raw targets jump by 10 px
        marker.set_path(path, np.array([positions[-1]-1., 0.]), 34, 0., .12, 50.)
        self.assertGreaterEqual(marker.update(.51, 0., .04)[0], positions[-1]-1e-10)

    def test_regeneration_projects_nearby_anchor_but_replaces_different_route(self):
        marker = self.marker([[0., 0.], [200., 0.]], [70., 0.])
        marker.set_path(np.array([[0., 2.], [200., 2.]]), [75., 2.], 4, 0., .12, 50.)
        np.testing.assert_array_equal(marker.point, [70., 2.])
        marker.set_path(np.array([[0., 200.], [200., 200.]]), [75., 200.], 7, 0., .12, 50.)
        np.testing.assert_array_equal(marker.point, [75., 200.])

    def test_endpoint_clipping_pause_and_reset(self):
        marker = self.marker([[0., 0.], [100., 0.]], [95., 0.], 200.)
        for i in range(1, 20):
            point = marker.update(i/120., 200., .04)
            self.assertLessEqual(point[0], 100.)
        np.testing.assert_array_equal(point, [100., 0.])
        np.testing.assert_array_equal(marker.update(20., 200., .04, paused=True), point)
        marker.set_path(np.array([[0., 0.], [40., 0.]]), [100., 0.], 4, 200., .12, 50.)
        np.testing.assert_array_equal(marker.point, [40., 0.])
        marker.reset()
        self.assertIsNone(marker.update(21., 200., .04))

    def test_first_waypoint_latch_stops_on_intersection_not_future_target(self):
        path=np.array([[0.,0.],[100.,0.],[100.,100.]])
        marker=self.marker(path,[90.,0.],speed=100.)
        marker.set_path(path,[100.,30.],4,100.,.12,50.,stop=[100.,0.])
        points=[marker.update(i/120.,100.,.04) for i in range(1,61)]
        self.assertTrue(marker.at_stop)
        np.testing.assert_array_equal(points[-1],[100.,0.])
        self.assertTrue(all(p[1]==0. and p[0]<=100. for p in points))
        for i in range(61,121):
            np.testing.assert_array_equal(marker.update(i/120.,100.,.04),[100.,0.])
        # Promotion removes the old cap; existing path interpolation resumes.
        marker.set_path(path,[100.,30.],7,100.,.12,50.,stop=[100.,100.])
        self.assertFalse(marker.at_stop)
        point=marker.update(1.01,100.,.04)
        self.assertGreater(point[1],0.)
        self.assertEqual(point[0],100.)

    def test_latched_marker_tracks_same_gap_intersection_after_nearby_regeneration(self):
        marker=self.marker([[0.,0.],[100.,0.],[200.,0.]],[150.,0.],speed=100.)
        marker.set_path(np.array([[0.,0.],[100.,0.],[200.,0.]]),[150.,0.],4,
                        100.,.12,50.,stop=[100.,0.])
        self.assertTrue(marker.at_stop)
        marker.set_path(np.array([[0.,1.],[101.,1.],[201.,1.]]),[150.,1.],7,
                        100.,.12,50.,stop=[101.,1.])
        self.assertTrue(marker.path_continuous)
        marker.hold_at_stop()
        np.testing.assert_array_equal(marker.update(.01,100.,.04),[101.,1.])
        marker.reset()
        self.assertFalse(marker.at_stop)
        self.assertIsNone(marker.stop_point)

    def test_large_route_change_discards_old_hold_anchor(self):
        marker=self.marker([[0.,0.],[100.,0.],[200.,0.]],[150.,0.],speed=100.)
        marker.set_path(np.array([[0.,0.],[100.,0.],[200.,0.]]),[150.,0.],4,
                        100.,.12,50.,stop=[100.,0.])
        marker.hold_at_stop()
        marker.set_path(np.array([[0.,200.],[100.,200.],[200.,200.]]),[70.,200.],7,
                        100.,.12,50.,stop=[100.,200.])
        self.assertFalse(marker.path_continuous)
        self.assertFalse(marker.at_stop)
        np.testing.assert_array_equal(marker.point,[70.,200.])

    def test_first_hold_does_not_clamp_raw_pursuit_or_control(self):
        from tests.test_heavy_motion_v2 import env_stub
        env,visuals=env_stub()
        obs=np.array([[4.,3.,.34],[4.,7.,.34],[7.,3.,.34],[7.,7.,.34]])
        env.navigation_map=SimpleNamespace(obstacles=obs)
        visuals.gui_clusters=list(obs[:,:2]*50.);visuals.gui_ids=list(range(4))
        env.boat_heading=0.;env.boat_vel=np.array([80.,0.]);env.dt=.04
        env.lidar_dists=np.full(180,320.);env.grid=np.zeros((4,4))
        visuals.gui_lidar_dists=env.lidar_dists.copy()
        env.renderer=SimpleNamespace(render=lambda *args:None)
        visuals._annotate(env)
        first,second=env.current_wp,env.next_wp
        env.boat_pos=first['pos']-np.array([59.,0.])
        visuals._annotate(env)
        route=env.control_path.copy();raw=env.pursuit_target.copy()
        commands=env.command_speed,env.command_yaw_rate
        with patch('heavy_motion_v2.time.perf_counter',side_effect=np.arange(31)*.02):
            for _ in range(31):visuals.render(env,(None,None))
        self.assertTrue(visuals.annotation_state.first_latched)
        np.testing.assert_array_equal(env.visual_pursuit_target,first['pos'])
        self.assertFalse(np.array_equal(raw,first['pos']))
        np.testing.assert_array_equal(env.pursuit_target,raw)
        np.testing.assert_array_equal(env.control_path,route)
        self.assertEqual((env.command_speed,env.command_yaw_rate),commands)
        env.boat_pos=first['pos']-np.array([41.,0.])
        visuals._annotate(env)
        self.assertEqual(env.current_wp['pair'],second['pair'])
        self.assertFalse(visuals.annotation_state.first_latched)

    def test_render_only_target_leaves_controller_paths_gaps_and_targets_intact(self):
        from tests.test_heavy_motion_v2 import env_stub
        env, visuals = env_stub()
        env.navigation_map = SimpleNamespace(obstacles=np.empty((0, 3)))
        env.trajectory_navigator = SimpleNamespace()
        env.boat_vel = np.array([80., 0.])
        env.dt = .04
        visuals._annotate(env)
        env.lidar_dists = np.full(180, 320.)
        env.grid = np.zeros((4, 4))
        visuals.gui_lidar_dists = env.lidar_dists.copy()
        env.renderer = SimpleNamespace(render=lambda *args: None)
        path, target, pursuit = (env.control_path.copy(), env.controller_target,
                                 env.pursuit_target.copy())
        bezier = env.bezier_path.copy()
        commands = env.command_speed, env.command_yaw_rate
        with patch('heavy_motion_v2.time.perf_counter', side_effect=[0., .01, .02]):
            for _ in range(3):
                visuals.render(env, (None, None))
        self.assertIsNotNone(env.visual_pursuit_target)
        np.testing.assert_array_equal(env.control_path, path)
        np.testing.assert_array_equal(env.bezier_path, bezier)
        np.testing.assert_array_equal(env.pursuit_target, pursuit)
        self.assertIs(env.controller_target, target)
        self.assertEqual((env.command_speed, env.command_yaw_rate), commands)
        visuals.reset_episode(env)
        self.assertIsNone(env.visual_pursuit_target)
        self.assertIsNone(visuals.pursuit_marker.point)


if __name__ == '__main__':
    unittest.main()
