"""MAIN compatibility and complete restoration of native mode state."""
import os
os.environ.setdefault('SDL_VIDEODRIVER','dummy')
os.environ.setdefault('SDL_AUDIODRIVER','dummy')
os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
import unittest
from unittest.mock import patch
import numpy as np
from environment import BoatEnv
import main_line_compat as compat
from experiments.evaluate_main_line_compat import NoRenderer, verify_pinned_sources, reference
from vessel_dynamics import allocate

class LineCompatibilityTests(unittest.TestCase):
    def setUp(self):
        with patch('environment.EnvRenderer',NoRenderer):self.env=BoatEnv()
    def tearDown(self):
        import pygame
        pygame.quit()
    def test_canonical_functions_have_no_source_drift(self):
        verify_pinned_sources()
    def test_profile_sensor_and_hull_match_main(self):
        ref=reference()[0]()
        env=self.env;env.set_line_tracing(True)
        for name in ('mass','inertia','drag','rot_drag','dt','lidar_beams','lidar_range'):
            self.assertEqual(getattr(env,name),getattr(ref,name))
        for name in ('rel_angles','left_hull_local','right_hull_local','deck_local'):
            np.testing.assert_array_equal(getattr(env,name),getattr(ref,name))
        for name in compat.MAIN_LINE_PARAMETERS:
            self.assertEqual(env.params[name],ref.params[name])
    def test_mode_switch_resets_velocity_and_actuator_state(self):
        env=self.env;native=env.dynamics;params=env.params.copy()
        env.boat_vel[:]=[50.,20.];env.boat_ang_vel=.5
        env.thrust_left=env.thrust_right=25.
        env.current_fwd=50.;env.prev_steer=1.
        env.set_line_tracing(True)
        self.assertEqual(env.line_physics_profile,compat.MAIN_LINE_PHYSICS)
        self.assertEqual((env.mass,env.inertia,env.drag,env.rot_drag,env.dt),(10,4.5,.2,.8,.04))
        np.testing.assert_array_equal(env.boat_vel,[0.,0.])
        self.assertEqual((env.boat_ang_vel,env.current_fwd,env.prev_steer,env.thrust_left,env.thrust_right),(0.,)*5)
        env.set_line_tracing(False)
        self.assertEqual(env.dynamics,native);self.assertEqual(env.params,params)
        self.assertEqual(env.mass,native.mass_kg);self.assertEqual(env.inertia,native.yaw_inertia_kg_m2)
        self.assertIsNone(env.line_physics_profile)
        self.assertFalse(hasattr(env,'drag'))
    def test_main_pwm_and_conversion_are_not_si_thrust(self):
        env=self.env;env.set_line_tracing(True)
        for steer in (-1.,-.1,0.,.1,1.):
            self.assertEqual(env.get_pwm(steer),compat.get_pwm(env,steer))
        self.assertEqual(env.pwm_to_thrust(1500),15000)
        self.assertEqual(env.get_pwm(0.),(1500,1500))
    def test_line_boundary_is_main_boundary(self):
        env=self.env;env.set_line_tracing(True)
        env.dynamic_obstacles=np.empty((0,3))
        for xy,collision in (([20.,100.],False),([17.,100.],True),([100.,20.],False),([100.,17.],True)):
            env.boat_pos=np.array(xy,dtype=np.float32)
            self.assertEqual(env.collide(),collision)
    def test_queued_line_selects_profile_before_first_step(self):
        self.env.linetrace_queued=True;self.env.reset()
        self.assertTrue(self.env.linetrace_mode)
        self.assertEqual(self.env.line_physics_profile,compat.MAIN_LINE_PHYSICS)
    def test_manual_mode_uses_native_physics_even_if_line_flag_saved(self):
        env=self.env;native=env.dynamics;env.set_line_tracing(True)
        env.manual_mode=True;env.reset()
        self.assertFalse(compat.active(env));self.assertIsNone(env.line_physics_profile)
        self.assertEqual(env.mass,native.mass_kg)
    def test_collision_and_arrival_parity(self):
        env=self.env;env.set_line_tracing(True)
        self.assertEqual(env.collide(),compat.collide(env))
        self.assertFalse(compat.goal_reached(70.));self.assertTrue(compat.goal_reached(69.999))
    def test_native_allocator_unchanged(self):
        env=self.env;env.motion_core_v2=True;env.command_speed=1.2
        for steer in (-.5,0.,.5):
            expected=allocate(env.physics_state(),1.2,steer*env.dynamics.max_yaw_rate_rad_s,env.dynamics)
            actual=[env.pwm_to_thrust(p) for p in env.get_pwm(steer)]
            np.testing.assert_allclose(actual,expected,atol=1e-14,rtol=0.)

if __name__=='__main__':unittest.main()
