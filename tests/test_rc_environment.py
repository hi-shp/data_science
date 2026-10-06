"""RC arena boundary, obstacle, and sensor-display behavior."""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import random
import unittest
from unittest.mock import patch

import numpy as np
import pygame

import environment
import ui_renderer


class Fake3D:
    def __init__(self, *args, **kwargs):
        self.hits = []

    def render(self, env, hits, width, height):
        self.hits.append(hits)
        return pygame.Surface((width, height))

    def start_render(self, env, hits, width, height):
        self.hits.append(hits)

    def finish_render(self, width, height):
        return pygame.Surface((width, height))


class RCEnvironmentTest(unittest.TestCase):
    def setUp(self):
        random.seed(2069)
        np.random.seed(2069)
        with patch.object(environment, "EnvRenderer", lambda env: None):
            self.env = environment.BoatEnv(headless=True)

    def tearDown(self):
        pygame.quit()

    def test_rc_entry_and_reset_keep_main_arena_layout(self):
        env = self.env
        self.assertEqual(env.boat_pos.tolist(), [65.0, env.sim_h / 2])
        self.assertEqual(env.target.tolist(), [env.map_w - 100.0, env.sim_h / 2])
        self.assertEqual((env.obs_n, env.obs_r, env.min_obs),
                         (int(80 * env.map_w / env.w), 17, 120))
        before = (env.fullscreen_3d, env.sim_speed)
        env.toggle_manual_mode()
        self.assertTrue(env.manual_mode)
        self.assertTrue(env.fullscreen_3d)
        self.assertEqual(env.sim_speed, 1)
        self.assertEqual(env.boat_pos.tolist(), [65.0, env.sim_h / 2])
        self.assertEqual(env.target.tolist(), [env.map_w - 100.0, env.sim_h / 2])
        env.reset_manual_episode()
        self.assertEqual(env.manual_collisions, 0)
        self.assertEqual(env.boat_pos.tolist(), [65.0, env.sim_h / 2])
        env.toggle_manual_mode()
        self.assertFalse(env.manual_mode)
        self.assertEqual((env.fullscreen_3d, env.sim_speed), before)

    def test_all_rc_edges_clamp_without_collision_and_autonomous_boundary_remains(self):
        env = self.env
        env.manual_mode = True
        env.dynamic_obstacles = np.empty((0, 3), dtype=np.float32)
        scale = env.dynamics.pixels_per_m
        for intended, expected in (
            ((-5., env.sim_h / 2), (25., env.sim_h / 2)),
            ((env.map_w + 5., env.sim_h / 2), (env.map_w - 25., env.sim_h / 2)),
            ((env.map_w / 2, -5.), (env.map_w / 2, 25.)),
            ((env.map_w / 2, env.sim_h + 5.), (env.map_w / 2, env.sim_h - 25.)),
        ):
            def outside(state, left, right, dt, params):
                result = np.array(state, copy=True)
                result[:2] = np.array(intended) / scale
                return result
            with patch.object(environment, "integrate", side_effect=outside):
                env.step(1500, 1500)
            np.testing.assert_allclose(env.boat_pos, expected)
            self.assertEqual(env.manual_collisions, 0)
            self.assertFalse(env.collide())
        env.manual_mode = False
        env.boat_pos = np.array([25., env.sim_h / 2])
        self.assertTrue(env.collide())

    def test_buoy_collision_still_counts_in_rc(self):
        env = self.env
        env.manual_mode = True
        env.boat_pos = np.array([env.map_w / 2, env.sim_h / 2])
        env.dynamic_obstacles = np.array([[*env.boat_pos, 17.]], dtype=np.float32)
        self.assertTrue(env.collide())
        env.step(1500, 1500)
        self.assertEqual(env.manual_collisions, 1)
        self.assertGreater(env.manual_collision_cooldown, 0)

    def test_rc_hides_2d_and_3d_lidar_but_autonomous_keeps_it(self):
        env = self.env
        with patch.object(ui_renderer, "Engine3D", Fake3D):
            renderer = ui_renderer.EnvRenderer(env)
        env.renderer = renderer
        env.fullscreen_3d = True
        env.show_lidar = True
        env.show_lidar_range = True
        hit_x = np.full(180, np.nan)
        hit_y = np.full(180, np.nan)
        hit_x[0], hit_y[0] = env.boat_pos[0] + 30, env.boat_pos[1]
        env.lidar_dists = np.zeros(180)

        env.manual_mode = True
        renderer.render(hit_x, hit_y)
        self.assertIsNone(renderer.engine_3d.hits[-1])
        first = pygame.Surface((env.w, env.sim_h))
        second = pygame.Surface((env.w, env.sim_h))
        renderer._draw_2d_world(hit_x, hit_y, lambda x: x, 0, target_surf=first)
        hit_x[0], hit_y[0] = env.boat_pos[0] + 70, env.boat_pos[1] + 20
        renderer._draw_2d_world(hit_x, hit_y, lambda x: x, 0, target_surf=second)
        self.assertTrue(np.array_equal(pygame.surfarray.array3d(first),
                                       pygame.surfarray.array3d(second)))
        manual_gauge = pygame.surfarray.array3d(renderer.cam_surf).copy()

        env.manual_mode = False
        renderer.render(hit_x, hit_y)
        self.assertIsInstance(renderer.engine_3d.hits[-1], tuple)
        auto_gauge = pygame.surfarray.array3d(renderer.cam_surf)
        red = np.array([230, 60, 50])
        self.assertGreater(np.all(auto_gauge == red, axis=2).sum(),
                           np.all(manual_gauge == red, axis=2).sum())
        third = pygame.Surface((env.w, env.sim_h))
        fourth = pygame.Surface((env.w, env.sim_h))
        renderer._draw_2d_world(hit_x, hit_y, lambda x: x, 0, target_surf=third)
        hit_x[0], hit_y[0] = env.boat_pos[0] + 30, env.boat_pos[1]
        renderer._draw_2d_world(hit_x, hit_y, lambda x: x, 0, target_surf=fourth)
        self.assertFalse(np.array_equal(pygame.surfarray.array3d(third),
                                        pygame.surfarray.array3d(fourth)))


if __name__ == "__main__":
    unittest.main()
