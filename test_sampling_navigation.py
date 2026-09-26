"""Safety/perception/model contracts for the isolated architecture prototypes."""
import unittest
import numpy as np
from experiments.sampling_navigation import SamplingNavigator, SamplingConfig, hull_clearance
import test_navigation_pipeline
from test_dynamics_tuning import configured
from vessel_dynamics import allocate, integrate


class SamplingTests(unittest.TestCase):
    def test_rollout_matches_execution_timestep_and_allocator(self):
        p = configured(); nav = SamplingNavigator(p)
        z = np.array([3., 6., .2, .5, -.1, .1, 3., 5.])
        u = np.array([[[1., .2], [.7, -.3], [0., 0.]]])
        states, _ = nav.rollout(z, u, np.empty((0,3)), 36., 12.6)
        for i, command in enumerate(u[0]):
            for _ in range(3):
                z = integrate(z, *allocate(z, *command, p), .04, p)
            np.testing.assert_allclose(states[0,i], z, atol=1e-13, rtol=0)

    def test_compiled_rollout_matches_reference_with_obstacles(self):
        p = configured(); nav = SamplingNavigator(p)
        rng = np.random.default_rng(731)
        state = np.array([5., 6., .2, .8, -.1, .1, 5., 6.])
        commands = rng.normal(size=(32, 12, 2))
        commands[:, :, 0] = np.clip(commands[:, :, 0]+1., -.375, 1.5)
        commands[:, :, 1] = np.clip(commands[:, :, 1]*.2, -.5, .5)
        obstacles = np.column_stack([rng.uniform(0, 12, 20),
                                     rng.uniform(0, 12, 20), np.full(20, .34)])
        reference = nav.rollout_reference(state, commands, obstacles, 36., 12.6)
        accelerated = nav.rollout(state, commands, obstacles, 36., 12.6)
        for before, after in zip(reference, accelerated):
            np.testing.assert_allclose(after, before, atol=1e-13, rtol=0)

    def test_unseen_world_cannot_change_either_prototype(self):
        sensor = test_navigation_pipeline.NavigationTests().sensor
        first, _ = sensor([[7.,5.,.35]])
        hidden, _ = sensor([[7.,5.,.35], [12.,6.,.4]])
        state = np.array([3.,6.,0.,0.,0.,0.,0.,0.])
        for mode in ['corridor', 'direct']:
            outputs = []
            for m in [first, hidden]:
                nav = SamplingNavigator(configured(), mode=mode,
                                        config=SamplingConfig(samples=48,horizon_steps=8))
                outputs.append(nav.plan(state, m, np.array([34.,6.]), 1))
            for a,b in zip(outputs[0],outputs[1]):
                np.testing.assert_array_equal(a,b)
        visible_a,_ = sensor([[7.,5.,.35]],position=(7.,6.))
        visible_b,_ = sensor([[7.,5.,.35],[12.,6.,.4]],position=(7.,6.))
        state[:2] = [7.,6.]
        paths=[]
        for m in [visible_a, visible_b]:
            nav = SamplingNavigator(configured())
            paths.append(nav.guidance(state,m,np.array([34.,6.]),1))
        self.assertFalse(np.array_equal(*paths))

    def test_hull_includes_stern_and_checks_all_candidates(self):
        z=np.zeros((2,8));z[:,:2]=[5.,6.];z[1,2]=np.pi
        obs=np.array([[4.3,6.22,.1]])
        clearance=hull_clearance(z,obs,36.,12.6)
        self.assertTrue(np.all(clearance<0))
        self.assertGreater(hull_clearance(z,np.empty((0,3)),36.,12.6).min(),3.)

    def test_commands_respect_existing_bounds(self):
        m,_=test_navigation_pipeline.NavigationTests().sensor([])
        nav=SamplingNavigator(configured(),mode='direct',config=SamplingConfig(samples=48,horizon_steps=8))
        command,states,_,_=nav.plan(np.array([3.,6.,0.,0.,0.,0.,0.,0.]),m,np.array([34.,6.]),1)
        self.assertLessEqual(abs(command[1]),.5)
        self.assertLessEqual(command[0],1.5)
        self.assertGreaterEqual(command[0],-.375)
        self.assertTrue(np.all(np.abs(states[:,6:])<=25.))

    def test_command_slew_bound_covers_first_action_and_future_knots(self):
        m,_=test_navigation_pipeline.NavigationTests().sensor([])
        nav=SamplingNavigator(configured(),mode='corridor',
                              config=SamplingConfig(samples=48,horizon_steps=8,yaw_command_step=.1))
        state=np.array([3.,6.,0.,0.,0.,0.,0.,0.])
        previous=0.
        for frame in [1,4]:
            command,_,_,_=nav.plan(state,m,np.array([34.,6.]),frame)
            self.assertLessEqual(abs(command[1]-previous),.10000000001)
            self.assertLessEqual(np.max(np.abs(np.diff(nav.sequence[:,1]))),.10000000001)
            for _ in range(3):
                state=integrate(state,*allocate(state,*command,configured()),.04,configured())
            previous=command[1]


if __name__=='__main__':unittest.main()
