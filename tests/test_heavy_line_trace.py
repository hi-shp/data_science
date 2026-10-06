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
from experiments.success_rate.evaluate_main_line_compat import NoRenderer, verify_pinned_sources, reference
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
    def test_live_switch_preserves_episode_and_maps_actuators(self):
        import random
        env=self.env;native=env.dynamics;params=env.params.copy()
        env.boat_vel[:]=[50.,20.];env.boat_ang_vel=.5
        env.thrust_left=25.;env.thrust_right=8.
        env.frame=123;env.manual_collisions=4
        env.raw_route=np.array([[1.,2.],[3.,4.]])
        preserved={k:getattr(env,k) for k in ('boat_pos','boat_vel','boat_heading','boat_ang_vel',
            'frame','obstacles','dynamic_obstacles','target','manual_start_time','manual_collisions','wakes','reflected_wakes','grid')}
        rng=random.getstate();np_rng=np.random.get_state()
        with patch.object(env,'reset',side_effect=AssertionError('episode reset')):
            env.set_line_tracing(True)
            self.assertEqual(env.line_physics_profile,compat.MAIN_LINE_PHYSICS)
            self.assertAlmostEqual(env.current_fwd,33*10*50/20)
            self.assertNotEqual(env.current_fwd,0.)
            self.assertIsNone(env.raw_route)
            for k,v in preserved.items():
                if isinstance(v,(np.ndarray,list)):self.assertIs(getattr(env,k),v)
                else:self.assertEqual(getattr(env,k),v)
            env.set_line_tracing(False)
        self.assertEqual(random.getstate(),rng)
        np.testing.assert_array_equal(np.random.get_state()[1],np_rng[1])
        np.testing.assert_allclose([env.thrust_left,env.thrust_right],[25.,8.],atol=1e-14,rtol=0)
        self.assertEqual(env.dynamics,native);self.assertEqual(env.params,params)
        self.assertEqual(env.mass,native.mass_kg);self.assertEqual(env.inertia,native.yaw_inertia_kg_m2)
        self.assertIsNone(env.line_physics_profile)
        self.assertFalse(hasattr(env,'drag'))
        self.assertTrue(env._line_resume_plan)
    def test_mode_switch_is_idempotent(self):
        env=self.env;env.set_line_tracing(True)
        env.prev_steer=.3;env.current_fwd=321.
        generation=env.line_mode_generation
        env.set_line_tracing(True)
        self.assertEqual((env.prev_steer,env.current_fwd,env.line_mode_generation),(.3,321.,generation))
    def test_cold_episode_still_initializes_main_memory(self):
        env=self.env;env.linetrace_mode=True;env.reset()
        self.assertEqual((env.current_fwd,env.thrust_left,env.thrust_right,env.prev_steer),(0.,)*4)
        self.assertEqual(env.boat_pos.dtype,np.float32)
    def test_return_maps_last_applied_main_output_and_respects_limits(self):
        env=self.env;env.set_line_tracing(True)
        env.step(1400,1600)
        self.assertEqual(env._main_line_applied_moment,2000*compat.MAIN_LINE_PARAMETERS['mom_coeff'])
        common=env.current_fwd/env.mass/env.dynamics.pixels_per_m*env.dynamics.mass_kg
        diff=env._main_line_applied_moment/env.inertia*.84*env.dynamics.yaw_inertia_kg_m2/env.dynamics.thruster_arm_m
        differential=float(np.clip(diff/2,-25,25))
        common=float(np.clip(common/2,-25+abs(differential),25-abs(differential)))
        expected=[common-differential,common+differential]
        state=env.physics_state()[:6].copy()
        env.set_line_tracing(False)
        np.testing.assert_array_equal(env.physics_state()[:6],state)
        np.testing.assert_allclose([env.thrust_left,env.thrust_right],expected)
    def test_first_line_step_uses_exact_main_equations_from_live_state(self):
        env=self.env
        env.boat_pos=np.array([105.1234567,315.3456789])
        env.boat_vel[:]=[51.,-4.];env.boat_heading=.23;env.boat_ang_vel=.17
        env.thrust_left=16.;env.thrust_right=8.;env.frame=121
        env.set_line_tracing(True);env.min_wide_dist=130.
        ref=reference()[0]();ref.linetrace_mode=True
        for k in ('boat_pos','boat_vel','boat_heading','boat_ang_vel','current_fwd','min_wide_dist','frame'):
            v=getattr(env,k);setattr(ref,k,v.copy() if isinstance(v,np.ndarray) else v)
        env.step(1450,1550);ref.step(1450,1550)
        np.testing.assert_array_equal(env.boat_pos,ref.boat_pos)
        np.testing.assert_array_equal(env.boat_vel,ref.boat_vel)
        self.assertEqual((env.boat_heading,env.boat_ang_vel,env.current_fwd),
                         (ref.boat_heading,ref.boat_ang_vel,ref.current_fwd))
    def test_first_return_step_uses_native_equations(self):
        from vessel_dynamics import integrate
        env=self.env;env.set_line_tracing(True)
        env.step(1460,1540);env.set_line_tracing(False)
        env.command_speed=1.;env.command_yaw_rate=.1
        left,right=env.get_pwm(.2)
        expected=integrate(env.physics_state(),env.pwm_to_thrust(left),env.pwm_to_thrust(right),env.dt,env.dynamics)
        env.step(left,right)
        np.testing.assert_allclose(env.physics_state(),expected,atol=1e-14,rtol=0)
    def test_reentry_plans_at_current_state_off_cadence(self):
        env=self.env
        import importlib.util
        if importlib.util.find_spec('heavy.motion_v2') is not None:
            from heavy.motion_v2 import HeavyMotionV2
            visuals=HeavyMotionV2(env)
            advance=visuals.advance
        else:
            from trajectory_runtime import advance_trajectory
            advance=advance_trajectory
        env.frame=4
        env.set_line_tracing(True);env.set_line_tracing(False)
        advance(env)
        self.assertEqual(env.frame,5)
        self.assertEqual(env.prediction_frame,5)
        self.assertIsNotNone(env.predicted_trajectory)
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
