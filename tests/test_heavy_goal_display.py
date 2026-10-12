"""Goal guide is safe, bounded and cannot mutate physical prediction/control."""
import numpy as np
import pytest
from heavy.goal_display import goal_guidance_path, corridor_clear
from heavy.motion_core.passage_geometry import physical_hull_polygons
from heavy.gap_annotation import clipped_display_route
from tests.test_heavy_gap_gui_semantics import gate

HULL = physical_hull_polygons()*50.
BOUNDS = (1000., 700.)
GOAL = np.array([900., 350.])


def guide(points, obs=()):
    path=np.asarray(points,dtype=float); original=path.copy()
    out, headings, connected=goal_guidance_path(path,np.zeros(len(path)),GOAL,
        np.asarray(obs).reshape(-1,3),HULL,10.,BOUNDS,50.)
    np.testing.assert_array_equal(path,original)
    assert len(out)==len(headings)
    if connected:
        np.testing.assert_array_equal(out[-1],GOAL)
        assert corridor_clear(out,np.asarray(obs).reshape(-1,3),HULL,10.,BOUNDS)
    return out,connected


@pytest.mark.parametrize('points',[
    [[400.,350.],[750.,350.],[960.,350.]], # overshoot
    [[500.,350.],[650.,350.]], # outside finite horizon
    [[600.,250.],[780.,270.],[840.,290.]], # oblique
    [[800.,350.]], # short
    [[800.,350.],[800.,350.]], # degenerate
    [[650.,350.],[750.,350.],[720.,250.]], # corner-directed tail
    [[900.,350.],[960.,350.]], # exact center
])
def test_safe_terminal_guide(points):
    _,connected=guide(points)
    assert connected


def test_blocked_connector_preserves_prediction():
    path=[[500.,350.],[550.,350.],[600.,360.]]
    out,connected=guide(path,[[700.,350.,17.]])
    assert not connected
    np.testing.assert_array_equal(out,path)


def test_side_wall_hull_envelope_rejects_unsafe_guidance():
    _,connected=guide([[800.,20.],[850.,50.]])
    assert not connected


def test_outside_terminal_zone_preserves_prediction():
    path=[[100.,350.],[200.,350.]]
    out,connected=guide(path)
    assert not connected
    np.testing.assert_array_equal(out,path)


def test_regeneration_keeps_same_goal_endpoint_without_backward_tail():
    for y in [350.,370.,330.]:
        out,connected=guide([[600.,350.],[800.,y],[960.,y]])
        assert connected
        assert np.all(out[:,0]<=GOAL[0]+1e-8)
        assert np.all(np.sum(np.diff(out,axis=0)*(GOAL-out[:-1]),axis=1)>=-1e-8)


def test_clipping_preserves_gap_before_goal_and_discards_gap_after_goal():
    path=np.array([[100.,240.],[500.,240.]])
    for first,second,goal,expected_first,expected_second in [
        (gate(200.),None,[450.,240.],True,False),
        (gate(400.),None,[250.,240.],False,False),
        (gate(200.),gate(400.,(2,3)),[450.,240.],True,True),
        (None,None,[250.,240.],False,False)]:
        a,b,f,s=clipped_display_route(path,first,second,np.array(goal),27.5)
        assert (f is not None)==expected_first
        assert (s is not None)==expected_second
        endpoint=b[-1] if b is not None else a[-1]
        np.testing.assert_array_equal(endpoint,second['pos'] if expected_second else
            first['pos'] if expected_first else goal)


def test_goal_beyond_sensor_support_is_not_declared_clear():
    path=np.array([[400.,350.],[500.,350.]])
    out,_,connected=goal_guidance_path(path,np.zeros(2),GOAL,np.empty((0,3)),
        HULL,10.,BOUNDS,50.,visible_radius=320.)
    assert not connected
    np.testing.assert_array_equal(out,path)


def test_unsafe_short_center_connector_is_not_drawn():
    path=np.array([[100.,240.],[500.,240.]])
    a,b,_,_=clipped_display_route(path,None,None,np.array([250.,250.]),27.5,
                                goal_connector_safe=lambda a,b:False)
    np.testing.assert_array_equal(a[-1],[250.,240.])
