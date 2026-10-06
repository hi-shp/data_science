"""V2 control/display isolation and perceived portal annotation checks."""
import math
import unittest
from types import SimpleNamespace

import numpy as np

from heavy_motion_v2 import HeavyMotionV2
from heavy_motion_core.controller_config import ControllerParameters
from heavy_motion_core.passage_geometry import physical_hull_polygons
from heavy_motion_core.passage_guidance import Passage
from heavy_motion_core.trajectory_modes import config_for_mode
from heavy_motion_core.perception import lidar_hits_np
from vessel_dynamics import VesselParameters


def env_stub():
    hull = physical_hull_polygons()*50.
    env = SimpleNamespace(
        dynamics=VesselParameters(), manual_mode=False, linetrace_mode=False,
        frame=1, map_w=600., sim_h=600., lidar_range=320.,
        boat_pos=np.array([100., 240.]), target=np.array([550., 240.]),
        left_hull_local=hull[0], right_hull_local=hull[1], deck_local=hull[2],
        physics_state=lambda: np.array([2., 4.8, 0., 1., 0., 0., 0., 0.]))
    visuals = HeavyMotionV2(env)
    env.prediction_frame = 1
    env.prediction_stride_steps = 3
    env.predicted_clearance = .4
    env.command_speed, env.command_yaw_rate = 1.5, .1
    env.control_path = np.array([[2., 4.8], [4., 4.8], [6., 4.8], [8., 4.8]])
    env.predicted_trajectory = env.control_path
    return env, visuals


class V2Tests(unittest.TestCase):
    def test_exact_forward_configuration_is_preserved(self):
        cfg = config_for_mode('eta_continuity_forward')
        self.assertTrue(cfg.forward_policy)
        self.assertTrue(cfg.exact_passage_safety)
        self.assertEqual(cfg.samples, 128)
        self.assertEqual(cfg.horizon_steps, 40)
        self.assertEqual(cfg.margin, .2)

    def test_actual_off_midpoint_crossing_is_annotation_not_target(self):
        env, visuals = env_stub()
        env.navigation_map = SimpleNamespace(obstacles=np.array([[6., 3., .34], [6., 7., .34]]))
        passage = Passage(np.array([6., 5.]), np.array([1., 0.]), 3.32, 1.48, 1.48, (0, 1))
        env.trajectory_navigator = SimpleNamespace(display_passages=(passage,))
        path = env.control_path.copy()
        visuals.gui_clusters = list(env.navigation_map.obstacles[:,:2]*env.dynamics.pixels_per_m)
        visuals.gui_ids = list(range(len(visuals.gui_clusters)))
        visuals._annotate(env)
        self.assertEqual(env.total_gaps_count, 1)
        self.assertAlmostEqual(env.current_wp['portal_s'], .45)
        np.testing.assert_array_equal(env.current_wp['pos'], [300., 240.])
        np.testing.assert_array_equal(env.control_path, path)
        self.assertEqual((env.command_speed, env.command_yaw_rate), (1.5, .1))

    def test_wall_portal_is_not_a_main_gui_gap(self):
        env, visuals = env_stub()
        env.navigation_map = SimpleNamespace(obstacles=np.array([[6., 8., .34]]))
        passage = Passage(np.array([6., 3.83]), np.array([1., 0.]), 7.66, 1.48, 1.48, (0,), 'bottom')
        env.trajectory_navigator = SimpleNamespace(display_passages=(passage,))
        visuals.gui_clusters = list(env.navigation_map.obstacles[:,:2]*env.dynamics.pixels_per_m)
        visuals.gui_ids = list(range(len(visuals.gui_clusters)))
        visuals._annotate(env)
        self.assertEqual(env.all_gaps, [])
        self.assertIsNone(env.current_wp)

    def test_portal_absent_when_actual_path_does_not_cross(self):
        env, visuals = env_stub()
        env.navigation_map = SimpleNamespace(obstacles=np.array([[6., 0., .34], [6., 2., .34]]))
        passage = Passage(np.array([6., 1.]), np.array([1., 0.]), 1.32, 1.48, 1.48, (0, 1))
        env.trajectory_navigator = SimpleNamespace(display_passages=(passage,))
        visuals.gui_clusters = list(env.navigation_map.obstacles[:,:2]*env.dynamics.pixels_per_m)
        visuals.gui_ids = list(range(len(visuals.gui_clusters)))
        visuals._annotate(env)
        self.assertIsNone(env.current_wp)

    def test_display_does_not_feed_back_into_control(self):
        env, visuals = env_stub()
        env.navigation_map = SimpleNamespace(obstacles=np.empty((0, 3)))
        sequence = np.array([[1.5, .1]])
        env.trajectory_navigator = SimpleNamespace(display_passages=(), sequence=sequence)
        env.controller_target = np.array([123., 456.])
        path, target = env.control_path.copy(), env.controller_target.copy()
        visuals.gui_clusters = list(env.navigation_map.obstacles[:,:2]*env.dynamics.pixels_per_m)
        visuals.gui_ids = list(range(len(visuals.gui_clusters)))
        visuals._annotate(env)
        visuals.prepare_display(env)
        np.testing.assert_array_equal(env.control_path, path)
        np.testing.assert_array_equal(env.controller_target, target)
        np.testing.assert_array_equal(env.trajectory_navigator.sequence, sequence)
        self.assertEqual((env.command_speed, env.command_yaw_rate), (1.5, .1))
        np.testing.assert_array_equal(env.bezier_path[0], env.boat_pos)
        self.assertIsNone(env.visual_selected_trajectory)
        self.assertIsNotNone(env.pursuit_target)
        self.assertTrue(env.show_all_gaps)

    def test_two_gaps_use_actual_route_crossings_not_midpoints(self):
        env, visuals = env_stub()
        env.navigation_map = SimpleNamespace(obstacles=np.array([
            [4., 3., .34], [4., 7., .34], [7., 3., .34], [7., 7., .34]]))
        env.trajectory_navigator = SimpleNamespace(display_passages=tuple(
            Passage(np.array([x, 5.]), np.array([1., 0.]), 3.32, 1.48, 1.48, pair)
            for x, pair in ((4., (0, 1)), (7., (2, 3)))))
        visuals.gui_clusters = list(env.navigation_map.obstacles[:,:2]*env.dynamics.pixels_per_m)
        visuals.gui_ids = list(range(len(visuals.gui_clusters)))
        visuals._annotate(env)
        np.testing.assert_allclose(env.current_wp['pos'], [200., 240.])
        self.assertIsNotNone(env.next_wp)
        for gap in (env.current_wp, env.next_wp):
            np.testing.assert_allclose(gap['pos'], gap['c1']+gap['portal_s']*(gap['c2']-gap['c1']))
            self.assertNotAlmostEqual(gap['portal_s'], .5)
            self.assertGreaterEqual(gap['portal_s'], gap['interval'][0])
            self.assertLessEqual(gap['portal_s'], gap['interval'][1])
        np.testing.assert_allclose(env.bezier_path[-1], env.current_wp['pos'])
        np.testing.assert_allclose(env.next_bezier_path[0], env.current_wp['pos'])
        np.testing.assert_allclose(env.next_wp['pos'], [350.,240.])
        self.assertLess(np.linalg.norm(env.next_bezier_path-env.next_wp['pos'],axis=1).min(),1e-8)

    def test_render_uses_cache_without_rebuilding_geometry(self):
        env, visuals = env_stub()
        env.navigation_map = SimpleNamespace(obstacles=np.empty((0, 3)))
        env.trajectory_navigator = SimpleNamespace(display_passages=())
        visuals.gui_clusters = list(env.navigation_map.obstacles[:,:2]*env.dynamics.pixels_per_m)
        visuals.gui_ids = list(range(len(visuals.gui_clusters)))
        visuals._annotate(env)
        cache = env.bezier_path
        for _ in range(5):
            visuals.prepare_display(env)
        self.assertIs(env.bezier_path, cache)

    def test_episode_reset_clears_main_wake_surfaces_and_predictions(self):
        from unittest.mock import patch
        from environment import BoatEnv
        import pygame
        class NullRenderer:
            def __init__(self,env):
                self.engine_3d=None
        with patch('environment.EnvRenderer',NullRenderer):
            env=BoatEnv()
        visuals=HeavyMotionV2(env)
        env.wakes,env.reflected_wakes=[[1.,2.,3.,4.]],[[1.,2.,3.,4.]]
        env.wake_surf.fill((255,255,255,255))
        env.path_surf.fill((255,255,255,255))
        env.trail.fill((255,255,255,255))
        env.current_wp={'pos':np.array([100.,200.])}
        env.next_wp={'pos':np.array([200.,300.])}
        env.control_path=np.ones((3,2))
        env.reset()
        for surface in (env.wake_surf,env.path_surf,env.trail):
            self.assertEqual(pygame.surfarray.array_alpha(surface).max(),0)
        self.assertEqual(env.wakes,[])
        self.assertEqual(env.reflected_wakes,[])
        self.assertIsNone(env.current_wp)
        self.assertIsNone(env.next_wp)
        self.assertIsNone(env.control_path)
        self.assertIsNone(env.bezier_path)
        self.assertIsNone(env.pursuit_target)
        self.assertEqual(env.all_gaps,[])
        self.assertIsNone(visuals.gui_lidar_dists)

    def test_numerical_worker_matches_authoritative_environment_step(self):
        from unittest.mock import patch
        from environment import BoatEnv
        from heavy_motion_worker import _Model, snapshot
        class NullRenderer:
            def __init__(self,env):
                self.engine_3d=None
        with patch('environment.EnvRenderer',NullRenderer):
            env=BoatEnv()
        HeavyMotionV2(env)
        model=_Model(snapshot(env))
        rng=np.random.default_rng(89)
        for left,right in rng.uniform(1100,1900,(80,2)):
            model.step(left,right)
            env.step(left,right)
            np.testing.assert_array_equal(model.physics_state(),env.physics_state())

    def test_default_renderer_is_exact_main_renderer(self):
        import ast, subprocess
        from pathlib import Path
        current = ast.parse(Path('ui_renderer.py').read_text())
        original = ast.parse(subprocess.check_output(['git','show','main_light:ui_renderer.py'],text=True))
        renderer = lambda tree: next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='EnvRenderer')
        class MainStyle(ast.NodeTransformer):
            def visit_IfExp(self,node):
                if (isinstance(node.test,ast.Call) and isinstance(node.test.func,ast.Name)
                    and node.test.func.id=='getattr' and len(node.test.args)>1
                    and isinstance(node.test.args[1],ast.Constant)
                    and node.test.args[1].value=='motion_core_v2'):
                    return node.orelse
                return self.generic_visit(node)
        self.assertEqual(ast.dump(MainStyle().visit(renderer(current))),ast.dump(renderer(original)))

    def test_main_candidate_population_contains_buoy_pairs_only(self):
        env, visuals = env_stub()
        visuals.gui_clusters = [np.array([300.,y]) for y in (50.,180.,300.,500.)]
        visuals.gui_clusters.append(np.array([50.,200.]))  # behind the bow
        visuals.gui_ids = list(range(5))
        env.navigation_map = SimpleNamespace(obstacles=np.array([[6.,y/50.,.34] for y in (50.,180.,300.,500.)]))
        env.trajectory_navigator = SimpleNamespace(display_passages=())
        visuals._annotate(env)
        self.assertEqual(env.total_gaps_count,6)
        self.assertEqual(len(env.all_gaps),6)
        self.assertTrue(all(all(i>=0 for i in gap['pair']) for gap in env.all_gaps))
        self.assertTrue(all('wall' not in gap for gap in env.all_gaps))
        for gap in env.all_gaps:
            np.testing.assert_array_equal(gap['pos'],(gap['c1']+gap['c2'])/2.)
        env.show_all_gaps=False
        visuals._annotate(env)
        self.assertEqual(env.all_gaps,[])
        self.assertEqual(env.total_gaps_count,6)

    def test_forward_crossing_cannot_invent_pair_absent_from_main_candidates(self):
        env,visuals=env_stub()
        obs=np.array([[1.6,2.8,.34],[6.,6.8,.34]])
        env.navigation_map=SimpleNamespace(obstacles=obs)
        visuals.gui_clusters=list(obs[:,:2]*50.);visuals.gui_ids=[0,1]
        # Its crossing is forward, but this pair does not exist in MAIN's
        # candidate pool (one endpoint is behind the bow).
        from heavy_gap_annotation import route_crossings
        event=route_crossings(env.control_path*50.,np.zeros(4),
            [dict(c1=obs[0,:2]*50.,c2=obs[1,:2]*50.,pair=(0,1))],
            obs*50.,physical_hull_polygons()*50.,10.,(600.,600.))
        self.assertTrue(event)
        visuals._annotate(env)
        self.assertEqual(env.total_gaps_count,0)
        self.assertIsNone(env.current_wp);self.assertIsNone(env.next_wp)

    def test_retained_pair_must_still_exist_in_current_display_candidate_set(self):
        env,visuals=env_stub()
        obs=np.array([[6.,3.,.34],[6.,7.,.34]])
        env.navigation_map=SimpleNamespace(obstacles=obs)
        visuals.gui_clusters=list(obs[:,:2]*50.);visuals.gui_ids=[0,1]
        visuals._annotate(env)
        self.assertIsNotNone(env.current_wp)
        visuals.gui_clusters=[];visuals.gui_ids=[]
        visuals._annotate(env)
        self.assertIsNone(env.current_wp);self.assertIsNone(env.next_wp)
        self.assertIsNone(visuals.annotation_state.current_first_gap)

    def test_persistent_pair_uses_current_candidate_endpoints_and_exact_crossing(self):
        env,visuals=env_stub()
        obs=np.array([[6.,3.,.34],[6.,7.,.34]])
        env.navigation_map=SimpleNamespace(obstacles=obs)
        visuals.gui_clusters=list(obs[:,:2]*50.);visuals.gui_ids=[0,1]
        visuals._annotate(env)
        visuals.gui_clusters=[p+np.array([5.,0.]) for p in visuals.gui_clusters]
        visuals._annotate(env)
        current=env.all_gaps[0]
        np.testing.assert_array_equal(env.current_wp['c1'],current['c1'])
        np.testing.assert_array_equal(env.current_wp['c2'],current['c2'])
        np.testing.assert_array_equal(env.current_wp['pos'],[305.,240.])
        self.assertEqual(visuals.annotation_state.first_gap_switch_count,0)
        self.assertFalse(any(g['pair']==env.current_wp['pair'] for g in env.candidate_wps))
        np.testing.assert_array_equal(env.bezier_path[-1],env.current_wp['pos'])

    def test_heading_turn_releases_persistent_rear_pair(self):
        env,visuals=env_stub()
        obs=np.array([[6.,3.,.34],[6.,7.,.34]])
        env.navigation_map=SimpleNamespace(obstacles=obs)
        visuals.gui_clusters=list(obs[:,:2]*50.);visuals.gui_ids=[0,1]
        env.boat_heading=0.;visuals._annotate(env)
        self.assertIsNotNone(env.current_wp)
        source=env.control_path.copy()
        env.boat_heading=np.pi;visuals._annotate(env)
        self.assertIsNone(env.current_wp);self.assertIsNone(env.next_wp)
        np.testing.assert_array_equal(source,env.control_path)

    def test_waypoint_invalidation_does_not_wait_for_next_prediction(self):
        from unittest.mock import patch
        env,visuals=env_stub()
        env.boat_heading=0.
        env.current_wp=dict(pos=np.array([50.,240.]))
        visuals.last_generation=env.prediction_frame
        with patch('heavy_motion_v2.advance_trajectory',return_value=(np.array([np.nan]),np.array([np.nan]))), \
             patch.object(visuals,'update_gui_scan'),patch.object(visuals,'_annotate') as annotate:
            visuals.advance(env)
        annotate.assert_called_once_with(env)
        self.assertEqual(env.prediction_frame,1)

    def test_main_completion_refreshes_between_prediction_ticks(self):
        from unittest.mock import patch
        env,visuals=env_stub()
        env.boat_heading=0.
        env.current_wp=dict(pos=np.array([139.,240.]))
        visuals.last_generation=env.prediction_frame
        before=env.control_path.copy()
        with patch('heavy_motion_v2.advance_trajectory',return_value=(np.array([np.nan]),np.array([np.nan]))), \
             patch.object(visuals,'update_gui_scan'),patch.object(visuals,'_annotate') as annotate:
            visuals.advance(env)
        annotate.assert_called_once_with(env)
        self.assertEqual(env.prediction_frame,1)
        np.testing.assert_array_equal(env.control_path,before)

    def test_completion_marks_visited_promotes_second_and_refreshes_clipping(self):
        env,visuals=env_stub()
        env.boat_heading=0.;env.visited=set()
        obs=np.array([[4.,3.,.34],[4.,7.,.34],[7.,3.,.34],[7.,7.,.34]])
        env.navigation_map=SimpleNamespace(obstacles=obs)
        visuals.gui_clusters=list(obs[:,:2]*50.);visuals.gui_ids=list(range(4))
        visuals._annotate(env)
        first,second=env.current_wp,env.next_wp
        self.assertIsNotNone(first);self.assertIsNotNone(second)
        raw=env.control_path.copy()
        env.boat_pos=first['pos']-np.array([59.,0.])
        visuals._annotate(env)
        self.assertEqual(env.current_wp['pair'],first['pair'])
        self.assertNotIn(first['pair'],env.visited)
        env.boat_pos=first['pos']-np.array([41.,0.])
        visuals._annotate(env)
        self.assertEqual(env.current_wp['pair'],second['pair'])
        self.assertEqual(visuals.annotation_state.switch_reason['first'],'promoted_second')
        self.assertIn(first['pair'],env.visited)
        self.assertIn(tuple(reversed(first['pair'])),env.visited)
        np.testing.assert_array_equal(env.bezier_path[-1],env.current_wp['pos'])
        np.testing.assert_array_equal(env.control_path,raw)

    def test_gui_lidar_matches_main_and_cannot_change_control(self):
        from perception import lidar_hits_np as main_lidar
        env, visuals = env_stub()
        angles=np.linspace(-math.pi,math.pi,180,endpoint=False)
        obstacles=np.array([[230.,240.,17.],[180.,280.,17.]])
        control,hx,hy=lidar_hits_np(env.boat_pos,0.,angles,obstacles,320.,(0.,0.,600.,600.))
        expected,ehx,ehy=main_lidar(env.boat_pos,0.,angles,obstacles,320.)
        env.lidar_dists=control
        env.grid=np.ones((5,5))
        visuals.update_gui_scan(env,(hx,hy))
        np.testing.assert_array_equal(visuals.gui_lidar_dists,expected)
        np.testing.assert_array_equal(visuals.gui_hits[0],ehx)
        np.testing.assert_array_equal(visuals.gui_hits[1],ehy)
        visuals.gui_grid=np.zeros_like(env.grid)
        captured=[]
        env.renderer=SimpleNamespace(render=lambda *hits:captured.append((env.lidar_dists,env.grid)))
        original_grid=env.grid
        visuals.render(env,(hx,hy))
        self.assertIs(env.lidar_dists,control)
        self.assertIs(env.grid,original_grid)
        self.assertIs(captured[0][0],visuals.gui_lidar_dists)
        self.assertIs(captured[0][1],visuals.gui_grid)

    def test_accelerated_gui_clusters_are_exact_main_results(self):
        from heavy_gap_display import MainDisplayClusters
        from perception import extract_clusters_from_grid
        rng = np.random.default_rng(409)
        cache = MainDisplayClusters()
        for _ in range(12):
            grid = np.zeros((70,180), dtype=np.float32)
            x, y = rng.integers(0,180,180), rng.integers(0,70,180)
            grid[y,x] = rng.uniform(.5,10.,180)
            for updated in (grid, grid*.99):
                actual, expected = cache.extract(updated), extract_clusters_from_grid(updated)
                np.testing.assert_array_equal(actual,expected)

    def test_reset_discards_previous_episode_controller_and_annotations(self):
        env, visuals = env_stub()
        env.navigation_map = SimpleNamespace()
        env.trajectory_navigator = SimpleNamespace()
        env.current_wp = {'pos': [10., 20.]}
        env.wakes, env.reflected_wakes = [[1,2,3,4]], [[5,6,7,8]]
        env.bezier_path, env.next_bezier_path = np.ones((3,2)), np.ones((3,2))
        env.pursuit_target, env.next_pursuit_target = np.ones(2), np.ones(2)
        visuals.last_frame = 700
        visuals.reset_episode(env)
        self.assertFalse(hasattr(env, 'navigation_map'))
        self.assertFalse(hasattr(env, 'trajectory_navigator'))
        self.assertIsNone(env.current_wp)
        self.assertIsNone(visuals.last_result)
        self.assertEqual(visuals.last_frame, -1)
        self.assertEqual(env.wakes, [])
        self.assertEqual(env.reflected_wakes, [])
        self.assertIsNone(env.next_wp)
        self.assertIsNone(env.bezier_path)
        self.assertIsNone(env.next_bezier_path)
        self.assertIsNone(env.pursuit_target)
        self.assertIsNone(env.next_pursuit_target)
        self.assertIsNone(env.predicted_trajectory)

    def test_control_wall_hits_are_hidden_from_main_gui_only(self):
        angles = np.linspace(-math.pi, math.pi, 180, endpoint=False)
        d, hx, hy = lidar_hits_np(np.array([100., 100.]), 0., angles,
                                np.empty((0, 3)), 320., (0., 0., 600., 600.))
        self.assertTrue(np.isfinite(hx[[0, 45]]).all())
        self.assertAlmostEqual(float(hx[0]), 0., places=4)
        self.assertAlmostEqual(float(hy[45]), 0., places=4)
        self.assertTrue(np.isnan(hx[90]))  # CODEX omits goal-facing xmax rays.
        env, visuals = env_stub()
        env.lidar_dists = d.copy()
        original = env.lidar_dists.copy()
        visuals.update_gui_scan(env, (hx, hy))
        self.assertTrue(np.isnan(visuals.gui_hits[0][[0,45]]).all())
        np.testing.assert_array_equal(env.lidar_dists, original)
        self.assertTrue((visuals.gui_lidar_dists[[0,45]] == env.lidar_range).all())


if __name__ == '__main__':
    unittest.main()

class PinnedMotionSourceTests(unittest.TestCase):
    def test_controller_sources_match_pinned_codex_except_display_exports(self):
        import ast
        import subprocess
        from pathlib import Path
        from heavy_motion_core import SOURCE_COMMIT
        root = Path(__file__).parent
        modules = [path for path in (root/'heavy_motion_core').rglob('*.py')
                   if path.name not in ('__init__.py', 'controller_config.py')]
        class Normalize(ast.NodeTransformer):
            def visit_Call(self, node):
                if isinstance(node.func, ast.Name) and node.func.id == 'njit':
                    node.keywords = [kw for kw in node.keywords if kw.arg != 'nogil']
                return self.generic_visit(node)
            def visit_ImportFrom(self, node):
                if node.module and node.module.startswith('heavy_motion_core.'):
                    node.module = node.module[len('heavy_motion_core.'):]
                return node
            def visit_If(self, node):
                # The only control-path extension is a fresh plan on explicit
                # live-mode re-entry; ordinary cadence remains pinned CODEX.
                if isinstance(node.test, ast.BoolOp) and ast.unparse(node.test.values[-1]) == "getattr(env, '_line_resume_plan', False)":
                    node.test = node.test.values[0]
                return self.generic_visit(node)
            def visit_Assign(self, node):
                names = {ast.unparse(target) for target in node.targets}
                if names & {'self.display_passages', 'env.motion_prediction_states', 'env._line_resume_plan'}:
                    return None
                return self.generic_visit(node)
        for path in modules:
            source_path = str(path.relative_to(root/'heavy_motion_core'))
            original = subprocess.check_output(
                ['git', 'show', f'{SOURCE_COMMIT}:{source_path}'], cwd=root, text=True)
            self.assertEqual(ast.dump(Normalize().visit(ast.parse(path.read_text()))),
                             ast.dump(ast.parse(original)), source_path)
