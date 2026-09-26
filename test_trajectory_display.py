"""Rendered prediction may lose its elapsed prefix; control data may not."""
import unittest
import numpy as np
from trajectory_display import future_trajectory


class VisualTrajectoryTests(unittest.TestCase):
    def test_elapsed_prefix_is_hidden_without_modifying_prediction(self):
        control = np.column_stack([np.arange(5.,dtype=float),np.zeros(5)])
        frozen=control.copy()
        path=future_trajectory(control,np.array([1.7,0.]),6)
        np.testing.assert_array_equal(control,frozen)
        np.testing.assert_array_equal(path[0],[1.7,0.])
        self.assertTrue(np.all(path[1:,0]>=2.))

    def test_render_uses_latest_position_between_controller_updates(self):
        control=np.column_stack([np.arange(5.,dtype=float),np.zeros(5)])
        a=future_trajectory(control,np.array([.1,0.]),1)
        b=future_trajectory(control,np.array([.3,0.]),2)
        self.assertEqual(a[0,0],.1)
        self.assertEqual(b[0,0],.3)
        self.assertFalse(np.array_equal(a,b))

    def test_expired_prediction_is_not_displayed_as_a_stale_route(self):
        control=np.column_stack([np.arange(5.,dtype=float),np.zeros(5)])
        path=future_trajectory(control,np.array([6.,0.]),20)
        np.testing.assert_array_equal(path,[[6.,0.]])

    def test_overtaken_segment_does_not_draw_backwards(self):
        control=np.column_stack([np.arange(6.,dtype=float),np.zeros(6)])
        path=future_trajectory(control,np.array([2.4,0.]),1)
        np.testing.assert_array_equal(path[0],[2.4,0.])
        self.assertTrue(np.all(path[1:,0]>=2.4))

    def test_self_crossing_uses_elapsed_time_not_old_branch(self):
        control=np.array([[0.,0.],[1.,0.],[2.,0.],[2.,1.],
                          [1.,1.],[0.,1.],[0.,0.],[-1.,0.]])
        path=future_trajectory(control,np.array([0.,.1]),18)
        np.testing.assert_array_equal(path[0],[0.,.1])
        self.assertTrue(np.all(path[1:,0]<=0.))


if __name__=='__main__':unittest.main()
