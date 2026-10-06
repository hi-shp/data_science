"""Regression for the default controller's observed wall passage."""

import unittest

import numpy as np

from experiments.sampling_navigation import SamplingNavigator
from navigation_map import NavigationMap
from tests.test_dynamics_tuning import configured
from trajectory_modes import config_for_mode


class ForwardWallPassageTests(unittest.TestCase):
    def test_default_forward_prediction_continues_through_safe_gap(self):
        obstacles = np.array([[5., 2.1, .3]])  # 1.8 m observed wall opening
        observed = NavigationMap(36., 12.6)
        observed.obstacles = obstacles
        goal = np.array([12., .9])
        base = np.array([2., .9, 0., .8, 0., 0., 5., 5.])

        for x in (2., 4.8, 5.5):
            with self.subTest(x=x):
                nav = SamplingNavigator(configured(), mode='trajectory_control',
                    config=config_for_mode('eta_continuity_forward'))
                state = base.copy()
                state[0] = x
                command, prediction, _, clearance = nav.plan(
                    state, observed, goal, 1)
                self.assertGreater(command[0], 0.)
                self.assertGreaterEqual(clearance, nav.cfg.margin)
                self.assertGreater(prediction[-1, 0], max(x, 5.3))

    def test_crosswise_hull_still_rejected(self):
        nav = SamplingNavigator(configured(), mode='trajectory_control',
            config=config_for_mode('eta_continuity_forward'))
        pose = np.array([[4., .9, np.pi / 2, 0., 0., 0., 0., 0.]])
        clearance = nav.safety_clearance(pose, np.array([[5., 2.1, .3]]),
                                         36., 12.6)
        self.assertLess(clearance[0], nav.cfg.margin)


if __name__ == '__main__':
    unittest.main()
