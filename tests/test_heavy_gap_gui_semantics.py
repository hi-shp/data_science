"""Display endpoint/passage policies and post-selection MAIN diagnostics."""
import unittest
from types import SimpleNamespace
import numpy as np
from heavy_gap_annotation import (select_route_gaps,clipped_display_route,segments_conflict)
from heavy_gap_diagnostics import compute_legacy_gap_metrics
from heavy_motion_core.passage_geometry import physical_hull_polygons
from tests.test_heavy_motion_v2 import env_stub


def gate(x,pair=(0,1)):
    return dict(c1=np.array([x,140.]),c2=np.array([x,340.]),pos=np.array([x,240.]),
        pair=pair,obstacle_pair=pair,route_arc=x-100.,route_segment=0,
        route_fraction=(x-100.)/400.,interval=(.27,.73),passage_extent=69.)


class GapGUISemanticsTests(unittest.TestCase):
    def setUp(self):
        self.path=np.array([[100.,240.],[500.,240.]])
        self.hull=physical_hull_polygons()*50.

    def test_one_gap_display_stops_at_actual_crossing(self):
        first=gate(200.)
        source=self.path.copy()
        before,after,a,b=clipped_display_route(self.path,first,None,np.array([550.,240.]),70.)
        np.testing.assert_array_equal(before[-1],first['pos'])
        self.assertIsNone(after);self.assertIsNone(b)
        np.testing.assert_array_equal(self.path,source)

    def test_two_gaps_clip_at_second_with_current_pursuit_only(self):
        env,visuals=env_stub()
        obs=np.array([[4.,3.,.34],[4.,7.,.34],[7.,3.,.34],[7.,7.,.34]])
        env.navigation_map=SimpleNamespace(obstacles=obs)
        env.trajectory_navigator=SimpleNamespace()
        visuals.gui_clusters=list(obs[:,:2]*50.);visuals.gui_ids=list(range(4))
        raw=env.control_path.copy()
        visuals._annotate(env)
        np.testing.assert_array_equal(env.bezier_path[-1],env.current_wp['pos'])
        np.testing.assert_array_equal(env.next_bezier_path[0],env.current_wp['pos'])
        np.testing.assert_array_equal(env.next_bezier_path[-1],env.next_wp['pos'])
        self.assertIsNone(env.next_pursuit_target)
        self.assertIsNotNone(env.pursuit_target)
        np.testing.assert_array_equal(env.control_path,raw)

    def test_no_gap_clips_exact_goal_instead_of_preserving_post_goal_knots(self):
        before,after,a,b=clipped_display_route(self.path,None,None,np.array([250.,240.]),70.)
        np.testing.assert_array_equal(before,[[100.,240.],[250.,240.]])
        self.assertIsNone(after)

    def test_goal_before_first_clips_center_and_hides_unreachable_markers(self):
        first=gate(400.)
        before,after,a,b=clipped_display_route(self.path,first,None,np.array([250.,241.]),70.)
        np.testing.assert_array_equal(before[-1],[250.,241.])
        self.assertIsNone(a);self.assertIsNone(b);self.assertIsNone(after)
        self.assertLessEqual(before[:,0].max(),250.)

    def test_goal_before_second_is_endpoint_even_after_first(self):
        before,after,a,b=clipped_display_route(self.path,gate(200.),gate(400.,(2,3)),np.array([300.,240.]),70.)
        np.testing.assert_array_equal(before[-1],[300.,240.])
        self.assertTrue(any(np.all(point==a['pos']) for point in before))
        self.assertIsNone(b);self.assertIsNone(after)

    def test_second_separation_responds_to_speed_and_actual_turn(self):
        events=[gate(x,(2*i,2*i+1)) for i,x in enumerate((200.,245.,300.,450.))]
        args=(events,self.path,np.zeros(2),self.hull)
        first,second=select_route_gaps(*args,0.,.37,10.)
        self.assertEqual(second['pos'][0],300.)
        first,second=select_route_gaps(*args,200.,1.,10.)
        self.assertEqual(second['pos'][0],450.)
        first,second=select_route_gaps(events,self.path,np.array([0.,3.2]),self.hull,0.,.37,10.)
        self.assertEqual(second['pos'][0],450.)

    def test_next_non_crossing_passage_is_found_beyond_x_crossing_region(self):
        first=gate(200.)
        crossed=gate(360.,(2,3))
        crossed['c1']=np.array([150.,150.]);crossed['c2']=np.array([500.,300.])
        self.assertTrue(segments_conflict(first,crossed,10.))
        far=gate(450.,(4,5))
        a,b=select_route_gaps([first,crossed,far],self.path,np.zeros(2),self.hull,0.,.37,10.)
        self.assertIs(b,far)
        a,b=select_route_gaps([first,crossed],self.path,np.zeros(2),self.hull,0.,.37,10.)
        self.assertIsNone(b)

    def test_legacy_metrics_match_main_at_midpoint_without_ranking(self):
        from navigation import find_gap
        from config import GRID_H,GRID_W
        g=gate(250.)
        obs=np.array([[250.,140.,17.],[250.,340.,17.]])
        boat=np.array([100.,240.]);target=np.array([550.,240.])
        original=find_gap([g['c1'],g['c2']],[0,1],boat,0.,target,set(),
                          np.zeros((GRID_H,GRID_W)),obs)
        self.assertIsNotNone(original)
        diagnostics=compute_legacy_gap_metrics(g,boat,0.,target,obs)
        for key,values in original['factors'].items():
            self.assertAlmostEqual(values['raw'],diagnostics['factors'][key]['raw'],places=10)
            self.assertEqual(values['w'],diagnostics['factors'][key]['w'])
        self.assertAlmostEqual(original['score'],diagnostics['score'],places=10)
        self.assertIsNone(compute_legacy_gap_metrics(None,boat,0.,target,obs))

    def test_diagnostic_weights_do_not_change_any_display_selection(self):
        env,visuals=env_stub()
        obs=np.array([[4.,3.,.34],[4.,7.,.34],[7.,3.,.34],[7.,7.,.34]])
        env.navigation_map=SimpleNamespace(obstacles=obs)
        env.trajectory_navigator=SimpleNamespace()
        visuals.gui_clusters=list(obs[:,:2]*50.);visuals.gui_ids=list(range(4))
        env.params={};visuals._annotate(env)
        before=[env.current_wp['pos'].copy(),env.next_wp['pos'].copy(),env.bezier_path.copy(),env.next_bezier_path.copy()]
        env.params=dict(align_exp=50.,heading_exp=0.,width_exp=500.,clear_exp=0.,fwd_exp=0.)
        visuals._annotate(env)
        after=[env.current_wp['pos'],env.next_wp['pos'],env.bezier_path,env.next_bezier_path]
        for a,b in zip(before,after):
            np.testing.assert_array_equal(a,b)
        self.assertEqual(env.current_wp['factors']['Width']['w'],500.)

if __name__=='__main__':
    unittest.main()
