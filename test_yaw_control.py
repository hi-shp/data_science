"""Physical inertia and controller torque authority are independent settings."""
import unittest
from dataclasses import replace
import numpy as np
from test_dynamics_tuning import configured
from vessel_dynamics import allocate, integrate


class YawAuthorityTests(unittest.TestCase):
    def test_fixed_gain_preserves_torque_when_inertia_changes(self):
        p = configured()
        z = np.zeros(8)
        a = allocate(z, 0., .3, replace(p, yaw_inertia_kg_m2=7.65))
        b = allocate(z, 0., .3, replace(p, yaw_inertia_kg_m2=3.8))
        np.testing.assert_array_equal(a, b)
        slow = integrate(z, *a, .04, replace(p, yaw_inertia_kg_m2=7.65))
        fast = integrate(z, *b, .04, replace(p, yaw_inertia_kg_m2=3.8))
        self.assertGreater(fast[5], slow[5]*1.8)

    def test_optional_gain_preserves_old_configuration_behavior(self):
        p = replace(configured(), yaw_rate_gain_Nm_s=None)
        z = np.zeros(8)
        left, right = allocate(z, 0., .2, p)
        moment = (right-left)*p.thruster_arm_m
        self.assertAlmostEqual(moment, p.yaw_inertia_kg_m2*.2/p.yaw_response_s)

    def test_response_is_faster_with_finite_actuator_lag(self):
        times = []
        peaks = []
        for inertia in [7.65, configured().yaw_inertia_kg_m2]:
            p = replace(configured(), yaw_inertia_kg_m2=inertia)
            z = np.zeros(8)
            rates = []
            for _ in range(150):
                left, right = allocate(z, 0., .5, p)
                z = integrate(z, left, right, .04, p)
                rates.append(z[5])
            times.append((np.flatnonzero(np.array(rates)>=.45)[0]+1)*.04)
            peaks.append(np.rad2deg(np.max(np.diff(np.r_[0.,rates]))/.04))
            self.assertLess(rates[0], .1)
            self.assertLess(max(rates), .51)
        self.assertLess(times[1], times[0]*.75)
        self.assertLess(peaks[1], 65.)

    def test_invalid_gain_is_rejected(self):
        for gain in [0., -1., float('nan'), float('inf')]:
            with self.assertRaises(ValueError):
                replace(configured(), yaw_rate_gain_Nm_s=gain)


if __name__ == '__main__':
    unittest.main()
