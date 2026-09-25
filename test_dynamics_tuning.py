"""Configured dynamics: physical speed equilibrium and bounded turn response."""
import json
import math
from pathlib import Path
from dataclasses import replace
import unittest
import numpy as np
from vessel_dynamics import VesselParameters, integrate, allocate


def configured():
    return VesselParameters(**json.loads(Path(__file__).with_name('vessel_config.json').read_text())['physics'])


def baseline():
    return replace(configured(), yaw_inertia_kg_m2=9., yaw_response_s=.85,
                   surge_quadratic_drag=18., cruise_speed_m_s=1., yaw_rate_gain_Nm_s=None)


def full_thrust(p, dt=.04, initial_speed=0.):
    z = np.zeros(8)
    z[3] = initial_speed
    speeds = []
    for _ in range(round(20/dt)):
        z = integrate(z, p.max_thrust_N, p.max_thrust_N, dt, p)
        speeds.append(z[3])
    return np.array(speeds)


def heading_step(p, dt=.04):
    z = np.zeros(8)
    headings, accelerations = [], []
    for _ in range(round(20/dt)):
        error = (math.pi/2-z[2]+math.pi)%(2*math.pi)-math.pi
        rate = np.clip(.65*error, -p.max_yaw_rate_rad_s, p.max_yaw_rate_rad_s)
        left, right = allocate(z, 0., rate, p)
        next_z = integrate(z, left, right, dt, p)
        accelerations.append((next_z[5]-z[5])/dt)
        headings.append(next_z[2])
        z = next_z
    return np.array(headings), np.array(accelerations)


class ConfiguredDynamicsTests(unittest.TestCase):
    def test_terminal_speed_follows_force_balance(self):
        p = configured()
        actual = full_thrust(p)[-1]
        drag = p.surge_linear_drag*actual+p.surge_quadratic_drag*actual**2
        self.assertAlmostEqual(drag, 2*p.max_thrust_N, places=7)
        self.assertAlmostEqual(actual/full_thrust(baseline())[-1], 1.5, places=6)

    def test_startup_force_is_not_scaled_with_terminal_speed(self):
        old, new = full_thrust(baseline()), full_thrust(configured())
        self.assertLess(new[0]/old[0], 1.05)
        self.assertLess(np.diff(np.r_[0.,new]).max()/.04, 2*configured().max_thrust_N/configured().mass_kg)

    def test_above_equilibrium_speed_decays_without_a_hard_clamp(self):
        p = configured()
        terminal = full_thrust(p)[-1]
        speed = full_thrust(p, initial_speed=terminal*1.2)
        self.assertGreater(speed[0], terminal)
        self.assertLess(speed[0], terminal*1.2)
        self.assertAlmostEqual(speed[-1], terminal, places=6)

    def test_turn_response_is_bounded_and_converges_with_timestep(self):
        old, _ = heading_step(baseline())
        new, accel = heading_step(configured())
        fine, _ = heading_step(configured(), .01)
        self.assertLess(np.rad2deg(np.max(np.abs(accel))), 60.)
        self.assertLess(new.max()-math.pi/2, old.max()-math.pi/2)
        self.assertLess(np.max(np.abs(new-fine[3::4])), np.deg2rad(1.))
        self.assertLess(abs(new[-1]-math.pi/2), np.deg2rad(.1))


if __name__ == '__main__':
    unittest.main()
