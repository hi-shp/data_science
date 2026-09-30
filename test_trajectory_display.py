"""Rendered prediction may lose its elapsed prefix; control data may not."""
import unittest
import numpy as np
from trajectory_display import future_trajectory, predicted_state_marker


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

    def test_predicted_marker_moves_fractionally_between_prediction_knots(self):
        prediction = np.column_stack([np.arange(15., dtype=float), np.zeros(15)])
        unchanged = prediction.copy()
        display = prediction.copy()
        positions = [predicted_state_marker(display, prediction_path=prediction,
                                            progress=phase)
                     for phase in (0., .25, .5, .75, 1.)]
        np.testing.assert_allclose(np.array(positions)[:, 0],
                                   [9., 9.25, 9.5, 9.75, 10.])
        np.testing.assert_array_equal(prediction, unchanged)
        np.testing.assert_array_equal(display, unchanged)

    def test_marker_interpolates_along_corner_not_across_it(self):
        path = np.array([[0., 0.], [1., 0.], [1., 1.], [2., 1.]])
        marker = predicted_state_marker(path, point_index=1,
                                        prediction_path=path, progress=.5)
        np.testing.assert_allclose(marker, [1., .5])

    def test_marker_stays_on_goal_clipped_display_path(self):
        prediction = np.column_stack([np.arange(15., dtype=float), np.zeros(15)])
        display = np.array([[0., 0.], [1., 0.], [6.5, 0.]])
        marker = predicted_state_marker(display, prediction_path=prediction,
                                        progress=.5)
        np.testing.assert_allclose(marker, [6.5, 0.])

    def test_nearby_replan_releases_anchor_without_backward_snap(self):
        path = np.column_stack([np.arange(15., dtype=float), np.zeros(15)])
        positions = [predicted_state_marker(path, prediction_path=path,
                                            progress=phase, anchor=[8.8, 0.])
                     for phase in (0., .5, 1.)]
        np.testing.assert_allclose(np.array(positions)[:, 0], [8.8, 9.4, 10.])
        changed = predicted_state_marker(path, prediction_path=path,
                                         progress=0., anchor=[8.8, 3.])
        np.testing.assert_allclose(changed, [9., 0.])


if __name__=='__main__':unittest.main()
