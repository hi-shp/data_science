import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ.setdefault('KABOAT_WIDTH', '1800')

from dataclasses import replace
import random
import numpy as np
import unittest
from unittest.mock import patch
from vessel_dynamics import VesselParameters, integrate


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(value) for name, value in globals().items()
                              if name.startswith('test_') and callable(value))


def run_force(p, left, right, dt=.04, duration=2., state=None):
    z = np.zeros(8) if state is None else state.copy()
    for _ in range(round(duration/dt)):
        z = integrate(z, left, right, dt, p)
    return z


def test_neutral_thrusters_do_not_accelerate():
    p = VesselParameters()
    np.testing.assert_array_equal(run_force(p, 0, 0), np.zeros(8))


def test_double_inertia_halves_initial_angular_acceleration():
    p = VesselParameters()
    a = integrate(np.zeros(8), -10, 10, .0001, p)
    b = integrate(np.zeros(8), -10, 10, .0001, replace(p, yaw_inertia_kg_m2=p.yaw_inertia_kg_m2/2))
    assert .49 < a[5]/b[5] < .51


def test_unpowered_vessel_dissipates_energy_without_instant_stop():
    p = VesselParameters()
    z = np.array([0., 0., 0., 1., .3, .4, 0., 0.])
    initial = .5*p.mass_kg*(z[3]**2+z[4]**2)+.5*p.yaw_inertia_kg_m2*z[5]**2
    result = run_force(p, 0, 0, duration=.04, state=z)
    energy = .5*p.mass_kg*(result[3]**2+result[4]**2)+.5*p.yaw_inertia_kg_m2*result[5]**2
    assert 0 < energy < initial
    assert 0 < result[5] < z[5]


def test_timestep_convergence_and_mirrored_turns():
    p = VesselParameters()
    coarse = run_force(p, 8, 15)
    fine = run_force(p, 8, 15, dt=.02)
    np.testing.assert_allclose(coarse[:6], fine[:6], atol=.02)
    mirror = run_force(p, 15, 8)
    np.testing.assert_allclose(coarse[[0,3]], mirror[[0,3]], atol=1e-10)
    np.testing.assert_allclose(coarse[[1,2,4,5]], -mirror[[1,2,4,5]], atol=1e-10)


def test_vectorized_predictor_matches_single_vessel():
    p = VesselParameters()
    a = integrate(np.zeros((2,8)), np.array([10.,-10.]), np.array([15.,8.]), .04, p)
    b = integrate(np.zeros(8), 10., 15., .04, p)
    np.testing.assert_allclose(a[0], b)


def test_reset_and_render_batch_size_do_not_change_physics():
    from environment import BoatEnv
    from simulation import advance
    states = []
    for batch in [1, 4, 8]:
        env = BoatEnv(headless=True)
        random.seed(1001)
        env.reset()
        for k in range(75):
            advance(env, k % batch, batch)
        states.append(env.physics_state())
        env.reset()
        assert env.thrust_left == env.thrust_right == env.boat_ang_vel == env.prev_steer == 0
    np.testing.assert_allclose(states[0], states[1], atol=1e-10)
    np.testing.assert_allclose(states[0], states[2], atol=1e-10)


def test_route_does_not_see_obstacles_outside_sensor_range():
    # The planner interface now requires an explicit sensor snapshot.
    from environment import BoatEnv
    from route_planner import route_target
    from navigation_map import NavigationMap
    env = BoatEnv(headless=True)
    env.frame = 1
    env.navigation_map = NavigationMap(env.map_w/50., env.sim_h/50.)
    env.dynamic_obstacles = np.array([[1200.,300.,17.]])
    first = route_target(env).copy()
    env.dynamic_obstacles = np.array([[1000.,400.,17.]])
    env.control_path = None
    second = route_target(env)
    assert len(env.navigation_map.obstacles) == 0
    np.testing.assert_array_equal(first,second)
    env.close()


def test_safety_capsule_encloses_collision_hull():
    from environment import BoatEnv
    env = BoatEnv(headless=True)
    vertices = np.array(env.left_hull_local+env.right_hull_local+env.deck_local)/50.
    dx = np.maximum.reduce([-.56-vertices[:,0],vertices[:,0]-.52,np.zeros(len(vertices))])
    dy = np.minimum(abs(vertices[:,1]-.22),abs(vertices[:,1]+.22))
    assert np.max(np.hypot(dx,dy)) <= .32 + 1e-9
    env.close()


def test_gui_accumulator_uses_wall_time_instead_of_one_step_per_frame():
    import pygame
    import main
    from environment import BoatEnv
    class Clock:
        def tick(self, fps):
            return 8  # 125 display updates per second
    for speed in [1,2,4]:
        env = BoatEnv(headless=True)
        env.clock = Clock()
        env.sim_speed = speed
        count = [0]
        def events():
            count[0] += 1
            return [pygame.event.Event(pygame.QUIT)] if count[0] > 125 else []
        with patch.object(main,'BoatEnv',return_value=env), patch.object(pygame.event,'get',side_effect=events):
            main.run()
        assert env.frame == 50*speed, (speed,env.frame)
