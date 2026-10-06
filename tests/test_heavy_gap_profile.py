"""Measured presentation cannot invent geometry or feed the control path."""
import unittest
from unittest.mock import patch
import numpy as np
from heavy_gap_profile import load_presentation_profile
from heavy_gap_annotation import local_passage_groups,select_route_gaps,route_crossings
from heavy_motion_core.passage_geometry import physical_hull_polygons
from tests.test_heavy_gap_gui_semantics import gate


class LegacyPresentationProfileTests(unittest.TestCase):
    def setUp(self):
        self.path=np.array([[100.,240.],[650.,240.]])
        self.hull=physical_hull_polygons()*50.
        self.profile=load_presentation_profile(50.)

    def compact(self,x,pair):
        g=gate(x,pair)
        g['c1']=np.array([x,170.]);g['c2']=np.array([x,310.])
        return g

    def choose(self,events):
        return select_route_gaps(local_passage_groups(events),self.path,np.zeros(2),
            self.hull,0.,.37,10.,self.profile)

    def test_too_close_first_is_skipped_for_typical_forward_crossing(self):
        close=self.compact(140.,(0,1));ahead=self.compact(236.,(2,3))
        first,_=self.choose([close,ahead])
        self.assertIs(first,ahead)

    def test_only_close_passage_remains_available_as_fallback(self):
        only=self.compact(150.,(0,1))
        first,second=self.choose([only])
        self.assertIs(first,only);self.assertIsNone(second)

    def test_new_first_does_not_use_crossing_inside_hull_footprint(self):
        only=self.compact(130.,(0,1))
        self.assertEqual(self.choose([only]),(None,None))

    def test_typical_distance_wins_over_very_far_vertical_gate(self):
        usual=self.compact(236.,(0,1));far=self.compact(500.,(2,3))
        usual['c1']+=np.array([-20.,0.]);usual['c2']+=np.array([20.,0.])
        first,_=self.choose([far,usual])
        self.assertIs(first,usual)

    def test_compact_width_precedes_verticality_within_local_region(self):
        wide=gate(230.,(0,1));wide['c1'][1]=40.;wide['c2'][1]=440.
        compact=self.compact(236.,(0,2))
        compact['c1']+=np.array([-20.,0.]);compact['c2']+=np.array([20.,0.])
        first,_=self.choose([wide,compact])
        self.assertIs(first,compact)

    def test_only_wide_real_passage_is_not_hidden(self):
        wide=gate(236.,(0,1));wide['c1'][1]=40.;wide['c2'][1]=440.
        first,_=self.choose([wide])
        self.assertIs(first,wide)

    def test_second_prefers_band_but_keeps_shorter_distinct_passage(self):
        first=self.compact(236.,(0,1))
        close=self.compact(306.,(2,3));far=self.compact(360.,(4,5))
        a,b=self.choose([first,close,far])
        self.assertIs(a,first);self.assertIs(b,far)
        a,b=self.choose([first,close])
        self.assertIs(b,close)

    def test_profile_length_units_scale_with_render_coordinates(self):
        doubled=load_presentation_profile(100.)
        for name,quantiles in self.profile.items():
            for q,value in quantiles.items():
                self.assertAlmostEqual(doubled[name][q],2.*value)

    def test_profile_never_resurrects_non_crossing_or_unsafe_gaps(self):
        obs=np.array([[236.,170.,17.],[236.,310.,17.],[240.,40.,17.],[400.,40.,17.]])
        gates=[dict(c1=obs[0,:2],c2=obs[1,:2]),dict(c1=obs[2,:2],c2=obs[3,:2])]
        events=route_crossings(self.path,np.zeros(2),gates,obs,self.hull,10.,(700.,600.))
        self.assertEqual(len(events),1)
        original=self.path.copy()
        with patch('navigation.find_gap',side_effect=AssertionError('global score')):
            first,_=select_route_gaps(events,self.path,np.zeros(2),self.hull,0.,.37,10.,self.profile)
        np.testing.assert_array_equal(first['pos'],[236.,240.])
        np.testing.assert_array_equal(self.path,original)
        unsafe=route_crossings(self.path,np.full(2,np.pi/2),gates,obs,self.hull,30.,(700.,600.))
        self.assertFalse(unsafe)
        self.assertEqual(self.choose(unsafe),(None,None))

if __name__=='__main__':
    unittest.main()
