"""Local presentation regularization must not choose a different passage."""
import unittest
import numpy as np
from heavy_gap_annotation import (local_passage_groups,select_route_gaps,
    route_crossings,route_bezier,clipped_display_route)
from heavy_motion_core.passage_geometry import physical_hull_polygons
from test_heavy_gap_gui_semantics import gate


class LocalPresentationTests(unittest.TestCase):
    def setUp(self):
        self.path=np.array([[100.,240.],[600.,240.]])
        self.headings=np.zeros(2)
        self.hull=physical_hull_polygons()*50.

    def choose(self,events):
        return select_route_gaps(local_passage_groups(events),self.path,
            self.headings,self.hull,0.,.37,10.)

    def diagonal(self,x,pair):
        g=gate(x,pair)
        # This diagonal truly intersects the route at g['pos'].
        g['c1']=np.array([x-40.,140.]);g['c2']=np.array([x+40.,340.])
        return g

    def test_equal_midpoint_distance_prefers_compact_local_representative(self):
        a=self.diagonal(200.,(0,1));vertical=gate(215.,(0,2))
        later=gate(460.,(3,4))
        groups=local_passage_groups([a,vertical,later])
        self.assertEqual(len(groups),2)
        first,second=self.choose([a,vertical,later])
        self.assertIs(first,vertical);self.assertIs(second,later)
        np.testing.assert_array_equal(first['pos'],[215.,240.])
        np.testing.assert_array_equal(self.path,[[100.,240.],[600.,240.]])

    def test_local_presentation_is_deterministic_under_candidate_reordering(self):
        a=self.diagonal(200.,(0,1));vertical=gate(215.,(0,2))
        later=gate(460.,(3,4))
        for events in ([a,vertical,later],[later,vertical,a],[vertical,a,later]):
            first,second=self.choose(events)
            self.assertIs(first,vertical);self.assertIs(second,later)

    def test_more_vertical_remote_passage_cannot_replace_first(self):
        a=self.diagonal(200.,(0,1));later=gate(460.,(2,3))
        first,second=self.choose([a,later])
        self.assertIs(first,a);self.assertIs(second,later)

    def test_near_unrelated_pairs_are_not_grouped_by_visual_proximity(self):
        a=self.diagonal(200.,(0,1));other=gate(215.,(2,3))
        self.assertEqual(len(local_passage_groups([a,other])),2)
        first,second=self.choose([a,other])
        self.assertIs(first,a);self.assertIsNone(second) # distinct pair, same crossing region

    def test_local_groups_do_not_grow_by_transitive_chaining(self):
        events=[gate(200.,(0,1)),gate(245.,(0,2)),gate(290.,(0,3))]
        groups=local_passage_groups(events)
        self.assertEqual(len(groups),2)
        self.assertEqual(groups[0]['passage_end_arc'],145.)
        first,second=self.choose(events)
        self.assertIsNone(second) # next group is still too close to this passage end

    def test_second_region_prefers_midpoint_proximity_over_x_appearance(self):
        first=gate(200.)
        crossed=gate(450.,(2,3));crossed['c1']=np.array([200.,340.])
        diagonal=self.diagonal(460.,(2,4))
        a,b=self.choose([first,crossed,diagonal])
        self.assertIs(a,first);self.assertIs(b,diagonal)

    def test_last_passage_representative_keeps_bezier_intersection(self):
        a=self.diagonal(200.,(0,1));vertical=gate(215.,(0,2))
        later=gate(460.,(3,4))
        first,second=self.choose([a,vertical,later])
        before,after,_,_=clipped_display_route(self.path,first,second,np.array([650.,240.]),70.)
        np.testing.assert_array_equal(before[-1],first['pos'])
        np.testing.assert_array_equal(after[0],first['pos'])
        np.testing.assert_array_equal(after[-1],second['pos'])
        # Whole unclipped source is unchanged by representative choice.
        curve,_=route_bezier(self.path,self.headings)
        self.assertTrue(np.all(curve[:,1]==240.))

    def test_exact_safe_duplicate_observed_pair_retains_presentation_options(self):
        obs=np.array([[300.,140.,17.],[300.,340.,17.]])
        candidates=[dict(c1=obs[0,:2]+[-10.,0.],c2=obs[1,:2]+[10.,0.]),
                    dict(c1=obs[0,:2].copy(),c2=obs[1,:2].copy()),
                    # Does not traverse this physical route: never eligible.
                    dict(c1=np.array([150.,140.]),c2=np.array([350.,140.]))]
        events=route_crossings(self.path,self.headings,candidates,obs,self.hull,10.,(700.,600.))
        self.assertEqual(len(events),1)
        self.assertEqual(len(events[0]['presentation_candidates']),2)
        first,_=select_route_gaps(events,self.path,self.headings,self.hull,0.,.37,10.)
        np.testing.assert_array_equal(first['c1'],obs[0,:2])
        np.testing.assert_array_equal(first['c2'],obs[1,:2])
        self.assertGreaterEqual(first['crossing_clearance'],10.)

    def test_midpoint_proximity_precedes_compactness_within_same_passage(self):
        compact=gate(200.);compact['c1'][1]=100.;compact['c2'][1]=310.
        balanced=gate(215.,(0,2));balanced['c1'][1]=100.;balanced['c2'][1]=390.
        first,_=self.choose([compact,balanced])
        self.assertIs(first,balanced) # 5 px from midpoint rather than 35 px
        np.testing.assert_array_equal(first['pos'],[215.,240.])
        self.assertFalse(np.array_equal(first['pos'],(first['c1']+first['c2'])/2.))

    def test_midpoint_differences_within_one_raster_pixel_use_compact_tie_break(self):
        compact=gate(200.);compact['c1'][1]+=.6;compact['c2'][1]+=.6
        wide=gate(215.,(0,2));wide['c1'][1]=90.;wide['c2'][1]=390.
        first,_=self.choose([wide,compact])
        self.assertIs(first,compact)

    def test_midpoint_priority_cannot_choose_a_remote_crossing_region(self):
        local=gate(200.);local['c1'][1]=100.;local['c2'][1]=310.
        remote=gate(500.,(2,3))
        first,_=self.choose([remote,local])
        self.assertIs(first,local)

    def test_second_uses_same_midpoint_rule_inside_its_own_future_region(self):
        first=gate(200.)
        asymmetric=gate(350.,(2,3));asymmetric['c1'][1]=50.;asymmetric['c2'][1]=300.
        balanced=gate(365.,(2,4));balanced['c1'][1]=130.;balanced['c2'][1]=340.
        distant=gate(550.,(5,6))
        a,b=self.choose([first,asymmetric,balanced,distant])
        self.assertIs(a,first);self.assertIs(b,balanced)
        np.testing.assert_array_equal(b['pos'],[365.,240.])
        self.assertFalse(np.array_equal(b['pos'],(b['c1']+b['c2'])/2.))

if __name__=='__main__':
    unittest.main()
