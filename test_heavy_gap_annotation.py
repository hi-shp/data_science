"""The selected physical route, not legacy scores, defines displayed GAPs."""
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from heavy_gap_annotation import route_bezier, route_crossings, split_display_route
from heavy_motion_core.passage_geometry import physical_hull_polygons
from test_heavy_motion_v2 import env_stub


class RouteAnnotationTests(unittest.TestCase):
    def crossings(self,path,obstacles,headings=None):
        points=np.asarray(path,dtype=float)
        obs=np.asarray(obstacles,dtype=float)
        candidates=[dict(c1=obs[a,:2],c2=obs[b,:2],pair=(a,b))
                    for a in range(len(obs)) for b in range(a+1,len(obs))]
        if headings is None:
            headings=np.zeros(len(points))
        return route_crossings(points,np.asarray(headings),candidates,obs,
            physical_hull_polygons()*50.,10.,(600.,600.))

    def test_route_order_is_independent_of_cluster_order_and_legacy_weights(self):
        env,visuals=env_stub()
        obs=np.array([[7.,3.,.34],[7.,7.,.34],[4.,3.,.34],[4.,7.,.34]])
        env.navigation_map=SimpleNamespace(obstacles=obs)
        env.trajectory_navigator=SimpleNamespace()
        env.params=dict(align_exp=10000.,width_exp=10000.,heading_exp=-10000.)
        visuals.gui_clusters=list(obs[:,:2]*50.);visuals.gui_ids=list(range(4))
        with patch('navigation.find_gap',side_effect=AssertionError('Legacy score called')):
            visuals._annotate(env)
        np.testing.assert_allclose(env.current_wp['pos'],[200.,240.])
        np.testing.assert_allclose(env.next_wp['pos'],[350.,240.])
        for gap in (env.current_wp,env.next_wp):
            self.assertTrue(gap['diagnostics_only'])
            self.assertIn('factors',gap)
            self.assertLess(min(np.linalg.norm(p-gap['pos'],axis=1).min()
                               for p in (env.bezier_path,env.next_bezier_path)),1e-8)
        self.assertLess(env.current_wp['route_arc'],env.next_wp['route_arc'])

    def test_no_virtual_second_gap_or_raw_route_fallback(self):
        env,visuals=env_stub()
        obs=np.array([[6.,3.,.34],[6.,7.,.34]])
        env.navigation_map=SimpleNamespace(obstacles=obs)
        env.trajectory_navigator=SimpleNamespace()
        visuals.gui_clusters=list(obs[:,:2]*50.);visuals.gui_ids=[0,1]
        visuals._annotate(env)
        self.assertIsNotNone(env.current_wp);self.assertIsNone(env.next_wp)
        self.assertLess(np.linalg.norm(env.bezier_path-env.current_wp['pos'],axis=1).min(),1e-8)
        env.control_path=np.array([[2.,4.8],[3.,4.8]])
        env.raw_route=np.array([[2.,4.8],[8.,4.8]])
        visuals._annotate(env)
        self.assertIsNone(env.current_wp);self.assertIsNone(env.next_wp)
        np.testing.assert_allclose(env.bezier_path[-1],[150.,240.])

    def test_recovery_route_is_unchanged_but_rear_gui_gap_is_empty(self):
        env,visuals=env_stub()
        obs=np.array([[1.5,2.8,.34],[1.5,6.8,.34]])
        env.navigation_map=SimpleNamespace(obstacles=obs)
        env.trajectory_navigator=SimpleNamespace()
        env.boat_heading=0.
        env.control_path=np.array([[2.,4.8],[1.,4.8]])
        env.motion_prediction_states=np.array([[1.,4.8,0.,-1.,0.,0.,0.,0.]])
        visuals.gui_clusters=list(obs[:,:2]*50.);visuals.gui_ids=[0,1]
        visuals._annotate(env)
        self.assertIsNone(env.current_wp);self.assertIsNone(env.next_wp)
        np.testing.assert_array_equal(env.control_path,[[2.,4.8],[1.,4.8]])
        np.testing.assert_allclose(env.bezier_path[-1],[50.,240.])
        self.assertEqual(env.total_gaps_count,0)  # MAIN's candidate display is unchanged.

    def test_true_knot_crossing_but_not_tangent_touch_or_gate_extension(self):
        obs=[[300.,150.,17.],[300.,350.,17.]]
        self.assertEqual(len(self.crossings([[100.,240.],[300.,240.],[400.,240.]],obs)),1)
        self.assertFalse(self.crossings([[100.,240.],[300.,240.],[100.,260.]],obs))
        self.assertFalse(self.crossings([[100.,240.],[300.,240.]],obs))
        self.assertFalse(self.crossings([[300.,240.],[400.,240.]],obs))
        self.assertFalse(self.crossings([[100.,450.],[400.,450.]],obs))

    def test_unsafe_crossing_and_crosswise_hull_rejected(self):
        obs=[[300.,180.,17.],[300.,300.,17.]]
        path=[[100.,240.],[400.,240.]]
        self.assertEqual(len(self.crossings(path,obs)),1)
        self.assertFalse(self.crossings(path,obs,[np.pi/2,np.pi/2]))
        self.assertFalse(self.crossings([[100.,205.],[400.,205.]],obs))

    def test_repeat_pair_and_nearby_shared_passage_are_one_event(self):
        obs=[[200.,150.,17.],[200.,350.,17.],[300.,150.,17.],[300.,350.,17.]]
        events=self.crossings([[100.,240.],[400.,240.]],obs)
        self.assertEqual(len(events),2)
        np.testing.assert_allclose([g['pos'][0] for g in events],[200.,300.])
        repeated=self.crossings([[100.,240.],[400.,240.],[100.,240.],[400.,240.]],obs[:2])
        self.assertEqual(len(repeated),1)

    def test_duplicate_clusters_mapped_to_same_geometry_are_not_double_counted(self):
        points=np.array([[100.,240.],[400.,240.]])
        obs=np.array([[300.,150.,17.],[300.,350.,17.]])
        gates=[dict(c1=obs[0,:2],c2=obs[1,:2],pair=(0,1)),
               dict(c1=obs[0,:2]+[.05,0.],c2=obs[1,:2]+[.05,0.],pair=(2,3))]
        events=route_crossings(points,np.zeros(2),gates,obs,physical_hull_polygons()*50.,10.,(600.,600.))
        self.assertEqual(len(events),1)

    def test_arc_length_not_vertex_index_determines_order(self):
        obs=[[200.,150.,17.],[200.,350.,17.],[350.,150.,17.],[350.,350.,17.]]
        events=self.crossings([[100.,240.],[199.,240.],[200.1,240.],[400.,240.]],obs)
        self.assertEqual(events[0]['route_arc'],100.)
        self.assertGreater(events[-1]['route_arc'],events[0]['route_arc'])

    def test_bezier_preserves_rollout_knots_and_bounds_visual_deviation(self):
        points=np.array([[100.,240.],[200.,240.],[260.,250.],[330.,280.]])
        curve,headings=route_bezier(points,np.array([0.,.1,.3,.4]))
        np.testing.assert_array_equal(curve[::4],points)
        for i in range(len(points)-1):
            linear=points[i]+(points[i+1]-points[i])*np.arange(4)[:,None]/4.
            self.assertLessEqual(np.linalg.norm(curve[4*i:4*i+4]-linear,axis=1).max(),.25)
        self.assertEqual(len(curve),len(headings))

    def test_gate_spanning_other_obstacles_is_not_a_single_passage(self):
        obs=np.array([[300.,80.,17.],[300.,180.,17.],[300.,300.,17.],[300.,440.,17.]])
        events=self.crossings([[100.,240.],[400.,240.]],obs)
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['obstacle_pair'],(1,2))

    def test_surface_cluster_offset_does_not_double_inflate_observed_circle(self):
        obs=np.array([[300.,140.,17.],[300.,340.,17.]])
        route=np.array([[100.,240.],[400.,240.]])
        hull=physical_hull_polygons()*50.
        candidates=[dict(c1=np.array([300.,180.]),c2=np.array([300.,300.]),pair=(0,1))]
        events=route_crossings(route,np.zeros(2),candidates,obs,hull,10.,(600.,600.))
        self.assertEqual(len(events),1)
        np.testing.assert_array_equal(events[0]['pos'],[300.,240.])
        self.assertGreaterEqual(events[0]['crossing_clearance'],10.)
        # Same display line with genuinely close observed buoys remains unsafe.
        close=np.array([[300.,215.,17.],[300.,265.,17.]])
        self.assertFalse(route_crossings(route,np.zeros(2),candidates,close,hull,10.,(600.,600.)))

if __name__=='__main__':
    unittest.main()
