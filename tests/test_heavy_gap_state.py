"""Display persistence releases using MAIN waypoint completion semantics."""
import unittest
import numpy as np
from heavy_gap_state import GapAnnotationState,crossed_portal,waypoint_completion_reason
from heavy_gap_annotation import (clipped_display_route,select_second_gap,second_gap_band,
                                  crossing_in_front,forward_crossings)
from heavy_motion_core.passage_geometry import physical_hull_polygons
from tests.test_heavy_gap_gui_semantics import gate


class PersistentGapTests(unittest.TestCase):
    def setUp(self):
        self.state=GapAnnotationState()
        self.hull=physical_hull_polygons()*50.
        self.path=np.array([[100.,240.],[600.,240.]])

    def update(self,events,frame=1,boat=None,valid=None,heading=None,generation=None):
        if boat is None:boat=self.path[0]
        if valid is None:valid={g['pair']:g for g in events}
        return self.state.update(events,self.path,np.zeros(2),self.hull,0.,.37,10.,None,
                                 np.asarray(boat),frame,lambda old:valid.get(old['pair']),heading=heading,generation=generation)

    def test_first_and_second_hold_identity_despite_new_prettier_candidates(self):
        a,b=gate(200.),gate(450.,(2,3))
        self.update([a,b])
        prettier=gate(170.,(4,5));other=gate(350.,(6,7))
        first,second=self.update([prettier,other,a,b],frame=4)
        self.assertIs(first,a);self.assertIs(second,b)
        self.assertEqual(self.state.first_gap_switch_count,0)
        self.assertEqual(self.state.second_gap_switch_count,0)
        self.assertEqual(self.state.diagnostics(4)['first_gap_age'],3)

    def test_new_narrower_candidates_cannot_evict_valid_first_or_second(self):
        a,b=gate(200.),gate(450.,(2,3))
        for gap in (a,b):
            gap['c1'][1]=40.;gap['c2'][1]=440.
        self.update([a,b])
        narrower_first=gate(215.,(0,4));narrower_second=gate(465.,(2,5))
        from unittest.mock import patch
        with patch('heavy_gap_state.select_route_gaps',side_effect=AssertionError('reranked first')), \
             patch('heavy_gap_state.select_second_gap',side_effect=AssertionError('reranked second')):
            first,second=self.update([a,narrower_first,b,narrower_second],frame=4)
        self.assertIs(first,a);self.assertIs(second,b)
        self.assertEqual(self.state.first_gap_switch_count,0)
        self.assertEqual(self.state.second_gap_switch_count,0)

    def test_actual_finite_crossing_promotes_existing_second(self):
        a,b,c=gate(200.),gate(450.,(2,3)),gate(550.,(4,5))
        self.update([a,b,c])
        self.path=np.array([[210.,240.],[600.,240.]])
        b=b.copy();b['route_arc']=240.
        c=c.copy();c['route_arc']=340.
        first,second=self.update([b,c],frame=4,boat=[210.,240.])
        self.assertIs(first,b);self.assertIs(second,c)
        self.assertEqual(self.state.switch_reason['first'],'promoted_second')
        self.assertEqual(self.state.first_gap_switch_count,1)

    def test_passed_gate_cannot_be_immediately_reacquired_from_a_looping_prediction(self):
        first=gate(200.)
        self.update([first])
        looping=first.copy();looping['route_arc']=120.
        selected,_=self.update([looping],frame=4,boat=[205.,240.],valid={first['pair']:looping})
        self.assertIsNone(selected)
        selected,_=self.update([looping],frame=7,boat=[210.,240.],valid={first['pair']:looping})
        self.assertIsNone(selected)
        # MAIN keeps the completed obstacle pair visited for the whole episode.
        selected,_=self.update([looping],frame=10,boat=[300.,240.],valid={first['pair']:looping})
        self.assertIsNone(selected)
        selected,_=self.update([looping],frame=13,boat=[300.,240.],valid={first['pair']:looping})
        self.assertIsNone(selected)
        self.assertIn((0,1),self.state.completed_identities)

    def test_main_proximity_releases_before_crossing_and_promotes_second(self):
        a,b=gate(200.),gate(450.,(2,3))
        self.update([a,b])
        first,_=self.update([a,b],frame=4,boat=[141.,240.])
        self.assertIs(first,b)
        self.assertEqual(self.state.switch_reason['first'],'promoted_second')
        self.assertIn((0,1),self.state.completed_identities)
        self.assertFalse(crossed_portal(np.array([100.,240.]),np.array([141.,240.]),a))
        self.assertFalse(crossed_portal(np.array([100.,500.]),np.array([300.,500.]),a))

    def test_main_proximity_boundary_is_strict_and_uses_dynamic_point(self):
        a=gate(200.)
        a['pos']=np.array([200.,255.]) # not the obstacle midpoint
        self.assertEqual(waypoint_completion_reason(a,[140.01,255.],0.),'proximity')
        self.assertIsNone(waypoint_completion_reason(a,[140.,255.],0.))
        self.assertIsNone(waypoint_completion_reason(a,[139.99,255.],0.))

    def test_main_gate_normal_and_lateral_boundaries(self):
        a=gate(200.)
        self.assertEqual(waypoint_completion_reason(a,[215.,340.],0.),'gate_passed')
        self.assertIsNone(waypoint_completion_reason(a,[214.99,340.],0.))
        self.assertEqual(waypoint_completion_reason(a,[259.99,340.],0.),'gate_passed')
        self.assertIsNone(waypoint_completion_reason(a,[260.,340.],0.))
        self.assertEqual(waypoint_completion_reason(a,[220.,359.99],0.),'gate_passed')
        self.assertIsNone(waypoint_completion_reason(a,[220.,360.],0.))

    def test_main_near_rear_completion_without_gate_geometry(self):
        a=dict(pos=np.array([0.,0.]))
        self.assertEqual(waypoint_completion_reason(a,[74.99,0.],0.),'behind_nearby')
        self.assertIsNone(waypoint_completion_reason(a,[75.,0.],0.))
        for degrees,expected in ((94.99,None),(95.01,'behind_nearby')):
            angle=np.deg2rad(degrees)
            boat=-70.*np.array([np.cos(angle),np.sin(angle)])
            self.assertEqual(waypoint_completion_reason(a,boat,0.),expected)

    def test_presentation_latch_holds_main_zone_until_hull_approaches(self):
        self.state=GapAnnotationState(completion_hull=self.hull)
        a,b=gate(200.),gate(450.,(2,3))
        self.update([a,b],heading=0.)
        self.state.latch_first(a)
        first,_=self.update([a,b],frame=4,boat=[141.,240.],heading=0.)
        self.assertIs(first,a)
        self.assertTrue(self.state.first_latched)
        self.assertNotIn((0,1),self.state.completed_identities)
        bow=float(np.max(self.hull[:,:,0]))
        first,_=self.update([a,b],frame=5,boat=[200.-bow,240.],heading=0.)
        self.assertIs(first,b)
        self.assertFalse(self.state.first_latched)
        self.assertEqual(self.state.switch_reason['first'],'promoted_second')

    def test_latch_releases_immediately_on_gate_crossing_away_from_waypoint(self):
        self.state=GapAnnotationState(completion_hull=self.hull)
        a,b=gate(200.),gate(450.,(2,3))
        self.update([a,b],boat=[180.,300.],heading=0.)
        self.state.latch_first(a)
        first,_=self.update([a,b],frame=4,boat=[201.,300.],heading=0.)
        self.assertIs(first,b)
        self.assertFalse(self.state.first_latched)
        self.assertIn((0,1),self.state.completed_identities)

    def test_route_invalidity_cancels_latch_without_marking_gap_complete(self):
        self.state=GapAnnotationState(completion_hull=self.hull)
        a=gate(200.);self.update([a],heading=0.)
        self.state.latch_first(a)
        first,_=self.update([],frame=4,valid={},heading=0.)
        self.assertIsNone(first)
        self.assertFalse(self.state.first_latched)
        self.assertNotIn((0,1),self.state.completed_identities)
        self.state.reset()
        self.assertFalse(self.state.first_latched)

    def test_invalid_first_hides_immediately_and_replacement_requires_confirmation(self):
        a=gate(200.);new=gate(260.,(2,3))
        self.update([a])
        first,_=self.update([new],frame=2,valid={new['pair']:new})
        self.assertIsNone(first)
        self.assertEqual(self.state.switch_reason['first'],'route_or_safety_invalid:confirmation_pending')
        first,_=self.update([new],frame=5,valid={new['pair']:new})
        self.assertIs(first,new)
        first,second=self.update([],frame=8,valid={})
        self.assertIsNone(first);self.assertIsNone(second)

    def test_crossing_position_is_exact_new_intersection_on_retained_identity(self):
        a=gate(200.)
        self.update([a])
        moved=a.copy();moved['pos']=np.array([200.,255.]);moved['portal_s']=.575
        first,_=self.update([moved],frame=4)
        self.assertEqual(first['pair'],a['pair'])
        np.testing.assert_array_equal(self.state.current_first_crossing,moved['pos'])
        self.assertEqual(self.state.first_gap_selected_frame,1)

    def test_changed_first_revalidates_second_order(self):
        a,b=gate(200.),gate(450.,(2,3));self.update([a,b])
        new=gate(500.,(4,5))
        first,second=self.update([new],frame=4,valid={new['pair']:new,b['pair']:b})
        self.assertIsNone(first);self.assertIsNone(second)
        first,second=self.update([new],frame=7,valid={new['pair']:new,b['pair']:b})
        self.assertIs(first,new);self.assertIsNone(second)

    def test_reset_clears_identity_age_counters_and_previous_episode_position(self):
        self.update([gate(200.),gate(450.,(2,3))])
        self.update([gate(450.,(2,3))],frame=4,boat=[141.,240.])
        self.assertTrue(self.state.completed_identities)
        self.state.reset()
        self.assertIsNone(self.state.current_first_gap)
        self.assertIsNone(self.state.current_second_crossing)
        self.assertIsNone(self.state.previous_boat)
        self.assertEqual(self.state.passed_annotations,[])
        self.assertEqual(self.state.completed_identities,set())
        self.assertEqual(self.state.diagnostics(20)['first_gap_age'],0)

    def test_goal_removes_hidden_persistent_markers(self):
        first,second=self.update([gate(200.),gate(450.,(2,3))])
        _,_,first,second=clipped_display_route(self.path,first,second,np.array([300.,240.]),70.)
        self.state.apply_visible(first,second,1)
        self.assertIsNotNone(self.state.current_first_gap)
        self.assertIsNone(self.state.current_second_gap)

    def test_exact_goal_segment_clipping_without_any_gap_or_last_point_replacement(self):
        source=np.array([[100.,240.],[200.,240.],[400.,240.],[550.,240.]])
        before=source.copy()
        path,_,a,b=clipped_display_route(source,None,None,np.array([300.,240.]),70.)
        np.testing.assert_array_equal(path,[[100.,240.],[200.,240.],[300.,240.]])
        np.testing.assert_array_equal(source,before)
        self.assertIsNone(a);self.assertIsNone(b)

    def test_no_goal_visit_preserves_unclipped_no_gap_path(self):
        path,_,_,_=clipped_display_route(self.path,None,None,np.array([400.,500.]),70.)
        np.testing.assert_array_equal(path,self.path)

    def test_x_second_and_nearly_coincident_crossing_are_not_new_annotations(self):
        first=gate(200.)
        cross=gate(300.,(2,3));cross['c1']=np.array([150.,140.]);cross['c2']=np.array([450.,340.])
        second=select_second_gap([first,cross],first,self.path,np.zeros(2),self.hull,0.,.37,10.)
        self.assertIsNone(second)
        duplicate=gate(201.,(4,5))
        self.assertIsNone(select_second_gap([first,duplicate],first,self.path,np.zeros(2),self.hull,0.,.37,10.))

    def test_second_prefers_back_of_bounded_band_not_farthest_visible_gate(self):
        from heavy_gap_profile import load_presentation_profile
        profile=load_presentation_profile(50.)
        first=gate(200.)
        near=gate(290.,(2,3));preferred=gate(340.,(4,5));far=gate(550.,(6,7))
        result=select_second_gap([first,near,preferred,far],first,self.path,
            np.zeros(2),self.hull,0.,.37,10.,profile)
        self.assertIs(result,preferred)
        extended=np.array([[100.,240.],[1000.,240.]])
        end=gate(950.,(8,9))
        self.assertIs(select_second_gap([first,near,preferred,far,end],first,
            extended,np.zeros(2),self.hull,0.,.37,10.,profile),preferred)
        a=second_gap_band([first],first,self.path,np.zeros(2),self.hull,0.,.37,10.,profile)
        b=second_gap_band([first],first,extended,np.zeros(2),self.hull,0.,.37,10.,profile)
        self.assertEqual(a['upper'],b['upper'])

    def test_new_further_horizon_crossing_does_not_replace_valid_second(self):
        a,b=gate(200.),gate(340.,(2,3))
        self.update([a,b])
        self.path=np.array([[100.,240.],[1000.,240.]])
        better_in_band=gate(365.,(4,5));end=gate(950.,(6,7))
        first,second=self.update([a,b,better_in_band,end],frame=4)
        self.assertIs(first,a);self.assertIs(second,b)
        self.assertEqual(self.state.second_gap_switch_count,0)
        self.assertEqual(self.state.switch_reason['second'],'retained')

    def test_retained_second_releases_if_route_crossings_collapse_together(self):
        a,b=gate(200.),gate(340.,(2,3));self.update([a,b])
        close=b.copy();close['route_arc']=a['route_arc']+20.
        replacement=gate(400.,(4,5))
        _,second=self.update([a,close,replacement],frame=4)
        self.assertIsNone(second)
        self.assertIn('confirmation_pending',self.state.switch_reason['second'])
        self.update([a,close,replacement],frame=7)
        _,second=self.update([a,close,replacement],frame=10)
        self.assertIs(second,replacement)

    def test_short_prediction_does_not_force_a_nearly_coincident_second(self):
        a,b=gate(200.),gate(230.,(2,3))
        self.path=np.array([[100.,240.],[245.,240.]])
        first,second=self.update([a,b])
        self.assertIs(first,a);self.assertIsNone(second)

    def test_only_x_or_too_close_second_candidates_leave_second_empty(self):
        first=gate(200.);near=gate(235.,(2,3));cross=gate(340.,(4,5))
        cross['c1']=np.array([150.,140.]);cross['c2']=np.array([450.,340.])
        second=select_second_gap([first,near,cross],first,self.path,
            np.zeros(2),self.hull,0.,.37,10.)
        self.assertIsNone(second)

    def test_persistence_tracks_exact_intersection_not_old_marker_coordinates(self):
        a,b=gate(200.),gate(340.,(2,3));self.update([a,b])
        moved=b.copy();moved['pos']=np.array([340.,250.]);moved['route_arc']=245.
        _,second=self.update([a,moved],frame=4)
        self.assertIs(second,moved)
        np.testing.assert_array_equal(self.state.current_second_crossing,moved['pos'])
        self.assertEqual(self.state.second_gap_selected_frame,1)

    def test_front_hemisphere_uses_exact_crossing_not_segment_midpoint(self):
        g=gate(200.);g['c1']=np.array([0.,140.]);g['c2']=np.array([200.,340.])
        g['pos']=np.array([50.,240.]) # behind, despite forward endpoint
        self.assertFalse(crossing_in_front(g,self.path[0],0.))
        self.assertEqual(forward_crossings([g],self.path[0],0.),[])
        g['pos']=np.array([100.,400.])
        self.assertTrue(crossing_in_front(g,self.path[0],0.)) # +90 degree
        g['pos']=np.array([100.,80.])
        self.assertTrue(crossing_in_front(g,self.path[0],0.)) # -90 degree
        self.assertFalse(crossing_in_front(g,self.path[0],np.pi/2))

    def test_rotating_bow_releases_both_annotations_even_if_route_is_still_valid(self):
        first,second=gate(200.),gate(340.,(2,3))
        self.update([first,second],heading=0.)
        first,second=self.update([first,second],frame=4,heading=np.pi)
        self.assertIsNone(first);self.assertIsNone(second)
        self.assertIsNone(self.state.current_second_gap)

    def test_rear_first_cannot_hide_front_representative_in_same_local_group(self):
        front=gate(200.);rear=gate(190.,(0,2));rear['pos']=np.array([50.,240.])
        group=rear.copy();group['presentation_candidates']=(rear,front)
        first,_=self.update([group],heading=0.)
        self.assertIs(first,front)

    def test_second_holds_when_new_first_group_member_extends_acquisition_region(self):
        from heavy_gap_annotation import local_passage_groups
        a,b=gate(200.),gate(320.,(2,3))
        self.update([a,b])
        alias=gate(260.,(0,5))
        groups=local_passage_groups([a,alias,b])
        first,second=self.update(groups,frame=4,valid={a['pair']:a,b['pair']:b})
        self.assertIs(first,a);self.assertIs(second,b)
        self.assertEqual(self.state.switch_reason['second'],'retained')
        self.assertEqual(self.state.second_gap_switch_count,0)
        self.assertGreater(self.state.second_preferred_band['minimum'],b['route_arc']-a['route_arc'])

    def test_small_midpoint_advantage_never_evicts_either_valid_identity(self):
        a,b=gate(200.),gate(340.,(2,3))
        for gap in (a,b):gap['c1'][1]+=3.;gap['c2'][1]+=3.
        self.update([a,b])
        new_first=gate(210.,(0,4));new_second=gate(350.,(2,5))
        first,second=self.update([a,new_first,b,new_second],frame=4)
        self.assertIs(first,a);self.assertIs(second,b)
        self.assertEqual(self.state.first_gap_switch_count,0)
        self.assertEqual(self.state.second_gap_switch_count,0)

    def test_pending_first_ignores_prettier_newcomer_and_repeated_generation(self):
        from unittest.mock import patch
        a,b,c=gate(200.),gate(250.,(2,3)),gate(260.,(4,5))
        self.update([a],generation=1)
        first,_=self.update([b],frame=4,generation=4)
        self.assertIsNone(first)
        first,_=self.update([b],frame=5,generation=4)
        self.assertIsNone(first)
        self.assertEqual(self.state.pending['first']['count'],1)
        with patch('heavy_gap_state.select_route_gaps',side_effect=AssertionError('reranked valid pending pair')):
            first,_=self.update([b,c],frame=7,generation=7)
        self.assertIs(first,b)

    def test_second_replacement_requires_one_more_complete_renewal_than_first(self):
        a,b,c=gate(200.),gate(340.,(2,3)),gate(450.,(4,5))
        self.update([a,b],generation=1)
        self.assertIsNone(self.update([a,c],frame=4,generation=4)[1])
        self.assertIsNone(self.update([a,c],frame=7,generation=7)[1])
        self.assertIs(self.update([a,c],frame=10,generation=10)[1],c)

    def test_temporarily_invalid_pair_is_hidden_and_same_current_pair_can_resume(self):
        a=gate(200.);self.update([a])
        first,_=self.update([],frame=4,valid={})
        self.assertIsNone(first)
        first,_=self.update([a,gate(210.,(2,3))],frame=7)
        self.assertIs(first,a)
        self.assertEqual(self.state.switch_reason['first'],'same_pair_resumed')
        self.assertEqual(self.state.identity_switches['first'],0)

    def test_switch_metrics_include_replacement_across_hidden_confirmation(self):
        a,b=gate(200.),gate(250.,(2,3))
        self.update([a]);self.update([b],frame=4)
        self.update([b],frame=7)
        metrics=self.state.diagnostics(10)
        self.assertEqual(metrics['first_gap_identity_switches_including_hidden'],1)
        self.assertAlmostEqual(metrics['average_first_gap_lifetime_sim_s'],.12)
        self.assertTrue(self.state.selection_events)
        self.state.reset()
        self.assertEqual(self.state.identity_switches,{'first':0,'second':0})
        self.assertEqual(self.state.pending,{'first':None,'second':None})
        self.assertEqual(self.state.selection_events,[])

    def test_no_pair_switch_is_needed_for_moving_crossing_on_same_identity(self):
        a=gate(200.);self.update([a])
        moved=a.copy();moved['pos']=np.array([200.,260.]);moved['route_arc']=110.
        first,_=self.update([moved],frame=4)
        self.assertIs(first,moved)
        self.assertEqual(self.state.identity_switches['first'],0)
        self.assertEqual(self.state.pending['first'],None)

if __name__=='__main__':unittest.main()
