"""Accelerators must preserve the frozen search and physical prediction."""
import unittest
import numpy as np
from route_planner import a_star, a_star_reference
from fast_constant_rollout import compiled_constant_rollout, parameter_vector
from fast_corridor import compiled_within_corridor
from test_dynamics_tuning import configured
from vessel_dynamics import allocate, integrate


class ExactAccelerationTests(unittest.TestCase):
    def test_astar_route_matches_reference_with_unknown_cost_and_obstacles(self):
        rng = np.random.default_rng(42)
        xs, ys = np.arange(144)*.25, np.arange(51)*.25
        for _ in range(10):
            free = rng.random((51,144)) > .08
            weight = 1. + 2.*rng.random((51,144))
            before = a_star_reference(free.copy(),weight,xs,ys,(25,5),(25,136),.25)
            after = a_star(free.copy(),weight,xs,ys,(25,5),(25,136),.25)
            if before is None:
                self.assertIsNone(after)
            else:
                np.testing.assert_array_equal(before,after)

    def test_corridor_check_agrees_with_dense_reference(self):
        if compiled_within_corridor is None:
            self.skipTest('optional accelerator unavailable')
        raw = np.array([[0.,0.],[1.,0.],[1.,1.]])
        starts, delta = raw[:-1], np.diff(raw,axis=0)
        length_sq = np.sum(delta*delta,axis=1)
        points = np.array([[.5,.1],[.9,.35],[1.3,.8],[1.41,.8]])
        for n in range(1,len(points)+1):
            sample=points[:n]
            offset=sample[:,None,:]-starts[None,:,:]
            frac=np.clip(np.sum(offset*delta,axis=2)/length_sq,0,1)
            distance=np.linalg.norm(offset-frac[:,:,None]*delta,axis=2)
            expected=bool(np.all(distance.min(axis=1)<=.4))
            self.assertEqual(expected,bool(compiled_within_corridor(sample,starts,delta,length_sq)))

    def test_constant_rollout_preserves_all_states_and_collision_cost(self):
        if compiled_constant_rollout is None:
            self.skipTest('optional accelerator unavailable')
        p=configured();rng=np.random.default_rng(17)
        state=np.array([4.,6.,.2,.8,-.1,.1,3.,5.])
        speeds=np.array([1.5,.7,0.,-.375]);rates=np.array([.1,-.2,.5,-.5])
        obstacles=np.column_stack([rng.uniform(2,12,15),rng.uniform(2,11,15),np.full(15,.34)])
        accelerated=compiled_constant_rollout(state,speeds,rates,obstacles,36.,12.6,.2,25,.2,parameter_vector(p))
        z=np.repeat(state[None,:],len(speeds),axis=0)
        closest=np.full(len(speeds),10.)
        blocked=np.zeros(len(speeds),dtype=bool)
        turn=np.zeros(len(speeds));history=[]
        for _ in range(25):
            z=integrate(z,*allocate(z,speeds,rates,p),.2,p)
            history.append(z.copy())
            dx=obstacles[None,:,0]-z[:,None,0];dy=obstacles[None,:,1]-z[:,None,1]
            c,s=np.cos(z[:,None,2]),np.sin(z[:,None,2])
            along,lateral=dx*c+dy*s,-dx*s+dy*c
            gap=np.maximum.reduce([-.56-along,along-.52,np.zeros_like(along)])
            side=np.minimum(abs(lateral-.22),abs(lateral+.22))
            margin=(np.hypot(gap,side)-obstacles[None,:,2]-.32).min(axis=1)
            closest=np.minimum(closest,margin);blocked|=margin<.2
            wall=np.minimum.reduce([z[:,1]-.93,12.6-.93-z[:,1],z[:,0]-.93,36.-.93-z[:,0]])
            closest=np.minimum(closest,wall);blocked|=wall<.05
            turn+=z[:,5]**2*.2
        expected=(z,closest,blocked,turn,np.stack(history,axis=1))
        for before,after in zip(expected,accelerated):
            np.testing.assert_array_equal(before,after)


if __name__=='__main__':unittest.main()
