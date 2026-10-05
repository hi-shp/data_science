"""Fixed-step ordering, reset isolation and unchanged worker results."""
import copy
import os
import random
import time
import unittest
from collections import deque
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')

import numpy as np

from environment import BoatEnv
from trajectory_runtime import advance_trajectory
from trajectory_worker import TrajectoryWorker


class TrajectoryWorkerTests(unittest.TestCase):
    def make_env(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        with patch('ui_renderer.Engine3D', side_effect=RuntimeError('test: no 3D')):
            env = BoatEnv(headless=True)
        env.navigation_mode = 'eta_continuity_forward'
        return env

    def wait_ready(self, worker):
        deadline = time.monotonic()+30.
        while not worker.available():
            self.assertLess(time.monotonic(), deadline, 'worker packet timeout')
            time.sleep(.001)

    def assert_tick(self, actual, reference, worker):
        expected_hits = advance_trajectory(reference)
        self.wait_ready(worker)
        hits = worker.advance(actual)
        np.testing.assert_array_equal(actual.physics_state(), reference.physics_state())
        np.testing.assert_array_equal(actual.grid, reference.grid)
        np.testing.assert_array_equal(hits, expected_hits)
        for name in ('lidar_dists', 'raw_route', 'control_path', 'controller_target'):
            np.testing.assert_array_equal(getattr(actual, name), getattr(reference, name))
        self.assertEqual(actual.command_speed, reference.command_speed)
        self.assertEqual(actual.command_yaw_rate, reference.command_yaw_rate)
        self.assertEqual(actual.prediction_frame, reference.prediction_frame)
        self.assertIs(actual.predicted_trajectory, actual.control_path)
        self.assertLessEqual(len(worker.ready), worker.capacity)

    def test_fixed_steps_pause_reset_and_resume_match_synchronous_pipeline(self):
        actual = self.make_env(2069)
        reference = self.make_env(2069)
        worker = TrajectoryWorker(actual, capacity=3)
        try:
            for _ in range(24):
                self.assert_tick(actual, reference, worker)
            # A queued prediction must not move authoritative physics while paused.
            state = actual.physics_state().copy()
            worker.available()
            np.testing.assert_array_equal(actual.physics_state(), state)
            actual = self.make_env(2000)
            reference = self.make_env(2000)
            worker.reset(actual)
            for _ in range(9):
                self.assert_tick(actual, reference, worker)
            self.assert_tick(actual, reference, worker)
            # Resume from a changed state with existing map/navigator memory, as
            # a temporary switch to the synchronous line-trace/manual path can do.
            # Frame 10 resumes at a NON-planning tick (11), retaining all state.
            actual.boat_heading += .03
            actual.boat_ang_vel = .02
            reference.boat_heading += .03
            reference.boat_ang_vel = .02
            worker.reset(actual)
            for _ in range(9):
                self.assert_tick(actual, reference, worker)
        finally:
            worker.close()
            worker.close()
            self.assertFalse(worker.process.is_alive())
            actual.close()
            reference.close()

    def test_wrong_tick_is_rejected_before_mutating_environment(self):
        worker = TrajectoryWorker.__new__(TrajectoryWorker)
        worker.ready = deque([{'frame': 99}])
        from types import SimpleNamespace
        env = SimpleNamespace(frame=10)
        with self.assertRaisesRegex(RuntimeError, 'different physics tick'):
            worker.advance(env)
        self.assertEqual(env.frame, 10)

    def test_snapshot_is_independent_of_gui_array_mutation(self):
        from trajectory_worker import snapshot
        from types import SimpleNamespace
        env = SimpleNamespace(frame=0, boat_pos=np.array([1., 2.]),
                              target=np.array([3., 4.]))
        values = snapshot(env)
        saved = copy.deepcopy(values)
        env.boat_pos[:] = -1.
        env.target[:] = 0.
        np.testing.assert_array_equal(values['boat_pos'], saved['boat_pos'])
        np.testing.assert_array_equal(values['target'], saved['target'])


if __name__ == '__main__':
    unittest.main()
