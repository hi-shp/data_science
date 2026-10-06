"""Terminal arrival constraints depend on measured vessel state."""
import math
import unittest
import numpy as np
from goal_guidance import approach_heading, goal_reached, terminal_cost


class TerminalGuidanceTests(unittest.TestCase):
    def test_heading_is_a_stable_final_tangent(self):
        goal = np.array([10., 6.])
        route = np.array([[2., 4.], [6., 4.], [9., 5.5], [10., 6.]])
        heading = approach_heading(route, np.array([5., 4.]), goal)
        self.assertIsNotNone(heading)
        # A later replan does not flip the terminal state as the boat approaches.
        changed = np.array([[9., 7.], [10., 6.]])
        self.assertEqual(approach_heading(changed, np.array([9., 6.]), goal, heading), heading)

    def test_reverse_and_high_yaw_rate_do_not_finish(self):
        goal = np.array([10., 6.]);heading=0.
        aligned = np.array([9.5, 6., 0., .7, 0., .02, 0., 0.])
        self.assertTrue(goal_reached(aligned, goal, heading))
        for index,value in [(2, math.pi), (3, -.3), (4, .5), (5, .25)]:
            bad=aligned.copy();bad[index]=value
            self.assertFalse(goal_reached(bad, goal, heading))

    def test_terminal_objective_penalizes_future_momentum(self):
        goal=np.array([10.,6.]);position=np.array([5.,6.])
        z=np.zeros((3,8));z[:,:2]=[9.2,6.];z[:,3]=.6
        z[1,5]=.3;z[2,2]=math.pi;z[2,3]=-.3
        cost=terminal_cost(z,position,goal,0.)
        self.assertLess(cost[0],cost[1])
        self.assertLess(cost[1],cost[2])
        self.assertTrue(np.allclose(terminal_cost(z,np.array([-3.,6.]),goal,0.),0.))


if __name__=='__main__':unittest.main()
