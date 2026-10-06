"""Physics and command-adapter checks for the experimental heavy GAP branch."""
import json
import math
import os
from pathlib import Path
import unittest
from unittest.mock import patch

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')

import numpy as np

from environment import BoatEnv
from vessel_dynamics import VesselParameters, allocate, integrate


class _NoRenderer:
    def __init__(self, env):
        self.engine_3d = None


class MainHeavyDynamicsTests(unittest.TestCase):
    def setUp(self):
        with patch('environment.EnvRenderer', _NoRenderer):
            self.env = BoatEnv()

    def test_environment_uses_the_authoritative_integrator(self):
        env = self.env
        commands = [(1900., 1900.)]*12 + [(1600., 1900.)]*12 + \
                   [(1500., 1500.)]*12 + [(1900., 1600.)]*12
        for frame, (left, right) in enumerate(commands, 1):
            before = env.physics_state()
            expected = integrate(before, env.pwm_to_thrust(left),
                                 env.pwm_to_thrust(right), env.dt, env.dynamics)
            env.frame = frame
            env.step(left, right)
            np.testing.assert_allclose(env.physics_state(), expected,
                                       rtol=0, atol=1e-11)

    def test_gap_command_uses_codex_allocator(self):
        env = self.env
        env.min_wide_dist = 999.
        steer = .4
        left, right = env.get_pwm(steer)
        expected_left, expected_right = allocate(
            env.physics_state(), env.command_speed,
            min(1.0, steer*env.params['yaw_command_gain'])*env.dynamics.max_yaw_rate_rad_s,
            env.dynamics)
        self.assertAlmostEqual(env.pwm_to_thrust(left), expected_left)
        self.assertAlmostEqual(env.pwm_to_thrust(right), expected_right)

    def test_manual_command_uses_the_same_allocator(self):
        env = self.env
        env.manual_throttle = .8
        env.manual_steer = -.5
        left, right = env.get_manual_pwm()
        expected_left, expected_right = allocate(
            env.physics_state(), .8*env.dynamics.cruise_speed_m_s,
            -.5*env.dynamics.max_yaw_rate_rad_s, env.dynamics)
        self.assertAlmostEqual(env.pwm_to_thrust(left), expected_left)
        self.assertAlmostEqual(env.pwm_to_thrust(right), expected_right)

    def test_maximum_thrust_terminal_speed(self):
        config = json.loads((Path(__file__).resolve().parents[1]/'vessel_config.json').read_text())
        p = VesselParameters(**config['physics'])
        z = np.zeros(8)
        for _ in range(1500):
            z = integrate(z, p.max_thrust_N, p.max_thrust_N, .04, p)
        force = 2*p.max_thrust_N
        exact = (-p.surge_linear_drag + math.sqrt(
            p.surge_linear_drag**2 + 4*p.surge_quadratic_drag*force)) / \
            (2*p.surge_quadratic_drag)
        self.assertLess(abs(z[3]-exact)/exact, .01)


if __name__ == '__main__':
    unittest.main()
