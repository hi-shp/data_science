"""MAIN's fixed-step playback must not depend on rendered frame count."""
import unittest
from unittest.mock import patch

import numpy as np
import pygame

import main
from playback_scheduler import playback_budget


class _Keys:
    def __getitem__(self, key):
        return key == pygame.K_w


class _Clock:
    def __init__(self, fps, wall_time):
        self.fps = fps
        self.wall_time = wall_time

    def tick(self, limit):
        self.wall_time[0] += 1.0 / self.fps


class _ManualEnv:
    def __init__(self, fps, wall_time):
        self.clock = _Clock(fps, wall_time)
        self.dt = 0.04
        self.sim_speed = 1
        self.manual_mode = True
        self.paused = False
        self.show_leaderboard = False
        self.fullscreen_3d = True
        self.obstacles = np.empty((0, 3))
        self.dynamic_obstacles = self.obstacles
        self.boat_pos = np.array([0.0, 0.0])
        self.boat_heading = 0.0
        self.boat_vel = np.zeros(2)
        self.target = np.array([10000.0, 0.0])
        self.rel_angles = np.empty(0)
        self.lidar_range = 320
        self.grid = np.zeros((2, 2))
        self.clusters = []
        self.cluster_ids = []
        self.frame = 0
        self.manual_throttle = 0.0
        self.manual_steer = 0.0
        self.steps = []
        self.renders = 0

    def update_dynamic_obstacles(self):
        pass

    def step(self, left, right, sub_step_idx=0, total_sub_steps=1):
        self.steps.append((self.manual_throttle, sub_step_idx, total_sub_steps))
        self.boat_pos[0] += self.manual_throttle * self.dt

    def update_camera(self):
        pass

    def render(self, hits_x, hits_y):
        self.renders += 1


class PlaybackTimingTest(unittest.TestCase):
    def run_manual(self, fps, frames, events=None, speed=1, speed_changes=None):
        wall_time = [0.0]
        env = _ManualEnv(fps, wall_time)
        env.sim_speed = speed
        count = [0]
        events = events or {}
        speed_changes = speed_changes or {}

        def get_events():
            count[0] += 1
            if count[0] > frames:
                return [pygame.event.Event(pygame.QUIT)]
            if count[0] in speed_changes:
                env.sim_speed = speed_changes[count[0]]
            return events.get(count[0], [])

        with patch.object(main, 'BoatEnv', return_value=env), \
             patch.object(main.time, 'perf_counter', side_effect=lambda: wall_time[0]), \
             patch.object(main.pygame.event, 'get', side_effect=get_events), \
             patch.object(main.pygame.key, 'get_pressed', return_value=_Keys()), \
             patch.object(main.pygame, 'quit'), \
             patch.object(main, 'lidar_hits_np', return_value=(np.empty(0), np.empty(0), np.empty(0))), \
             patch.object(main, 'update_grid'), \
             patch.object(main, 'extract_clusters_from_grid', return_value=[]), \
             patch.object(main, 'match_clusters', return_value=([], [])):
            main.run()
        return env

    def test_one_second_rc_progress_is_independent_of_120_60_30_fps(self):
        results = [self.run_manual(fps, fps) for fps in (120, 60, 30)]
        for fps, env in zip((120, 60, 30), results):
            self.assertLessEqual(abs(len(env.steps) - 120), 1)
            self.assertEqual(env.renders, fps)
            self.assertTrue(all(idx == 0 and total == 1
                                for _, idx, total in env.steps))
        n = min(len(env.steps) for env in results)
        reference = [item[0] for item in results[0].steps[:n]]
        for env in results[1:]:
            np.testing.assert_allclose([item[0] for item in env.steps[:n]],
                                       reference, atol=0, rtol=0)
            self.assertLessEqual(abs(env.boat_pos[0] - results[0].boat_pos[0]), 0.05)

    def test_speed_change_discards_old_debt(self):
        self.assertEqual(playback_budget(3.0, 5.0, 8, 1, main.BASE_PLAYBACK_RATE), 0.0)
        self.assertEqual(playback_budget(0.02, 0.1, 1, 1, main.BASE_PLAYBACK_RATE), 0.5)

    def test_displayed_multipliers_scale_fixed_step_budget(self):
        for speed, requested_steps in ((2, 240), (4, 480)):
            env = self.run_manual(60, 60, speed=speed)
            self.assertLessEqual(abs(len(env.steps) - requested_steps), 2)

    def test_pause_does_not_accumulate_catch_up_steps(self):
        space = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE)
        env = self.run_manual(60, 60, {21: [space], 41: [space]})
        # Forty active render intervals request about eighty fixed steps.
        self.assertGreaterEqual(len(env.steps), 76)
        self.assertLessEqual(len(env.steps), 82)

    def test_high_speed_debt_does_not_spill_into_lower_speed(self):
        env = self.run_manual(60, 30, speed=8, speed_changes={21: 1})
        # First 20 frames hit the 8-step cap; after the change only the
        # new 1x wall budget may run, rather than the old 8x backlog.
        self.assertGreaterEqual(len(env.steps), 175)
        self.assertLessEqual(len(env.steps), 185)


if __name__ == '__main__':
    unittest.main()
