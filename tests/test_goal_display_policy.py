"""Frame-by-frame goal presentation transitions must never bypass safety."""
import numpy as np
from goal_display_policy import TerminalDisplayState, safe_goal_path, goal_prefix
from passage_geometry import physical_hull_polygons

HULL=physical_hull_polygons()*50.
GOAL=np.array([900.,350.]);BOUNDS=(1000.,700.)


def update(state,frame=1,safe=True,path=None):
    path=np.array([[750.,350.],[950.,350.]]) if path is None else path
    def connect(p,a):
        if safe:return np.vstack((p[0],GOAL)),np.zeros(2),True,'GOAL_CONNECTED'
        return p.copy(),a.copy(),False,'GOAL_OUTSIDE_RANGE'
    return state.update(path,np.zeros(len(path)),GOAL,frame,600.,70.,connect,lambda p:p.copy())


def test_goal_connected_stays_exact_across_new_predictions():
    state=TerminalDisplayState()
    for frame in range(10):
        p,_,valid=update(state,frame,path=np.array([[750.+frame,350.],[940.,330.+frame]]))
        assert valid
        np.testing.assert_allclose(p[-1],GOAL,atol=1e-6,rtol=0)


def test_one_frame_uncertain_sensor_does_not_reveal_corner_tail():
    state=TerminalDisplayState();update(state)
    p,_,valid=update(state,2,False,np.array([[760.,350.],[950.,60.]]))
    assert not valid and len(p)==1
    assert state.reason=='GUIDANCE_SUPPRESSED_GOAL_OUTSIDE_RANGE'
    p,_,valid=update(state,3)
    assert valid
    np.testing.assert_array_equal(p[-1],GOAL)


def test_real_blocker_does_not_hold_previous_safe_connector():
    state=TerminalDisplayState();update(state)
    for f in (2,3,4):
        p,_,valid=update(state,f,False)
        assert not valid and len(p)==1


def test_reset_does_not_carry_terminal_or_safe_state():
    state=TerminalDisplayState();update(state,20)
    update(state,1,False)
    assert not state.ever_connected


def test_finite_post_goal_tail_is_removed_before_connection():
    state=TerminalDisplayState()
    p,_,valid=update(state,safe=False,path=np.array([[750.,350.],[910.,350.],[950.,50.]]))
    assert state.terminal and not valid
    np.testing.assert_array_equal(p[-1],GOAL)
    # Exact physical intersection is allowed, but is not called certified guidance.
    assert len(p)==2



def test_curved_certified_prediction_keeps_needed_avoidance():
    # The chord is blocked, but a real, safety-certified selected rollout bends
    # around it. Only the short final connector is additional guidance.
    path=np.array([[650.,350.],[700.,250.],[820.,250.],[895.,350.]])
    obs=np.array([[760.,350.,17.]])
    original=path.copy()
    p,_,valid=safe_goal_path(path,np.zeros(4),GOAL,obs,HULL,10.,BOUNDS,50.,
        visible_radius=400.,prediction_clearance=.4)
    assert valid
    np.testing.assert_array_equal(p[-1],GOAL)
    np.testing.assert_array_equal(p[1:3],path[1:3])
    np.testing.assert_array_equal(path,original)
    _,_,unsafe=safe_goal_path(path,np.zeros(4),GOAL,obs,HULL,10.,BOUNDS,50.,
        visible_radius=400.,prediction_clearance=.1)
    assert not unsafe


def test_float32_si_conversion_is_not_used_for_exact_world_goal():
    state=TerminalDisplayState()
    world=np.array([1740.,322.],dtype=np.float32)
    converted=np.asarray(world/50.,dtype=float)*50.
    assert not np.allclose(converted,world,atol=1e-6,rtol=0)
    p=np.array([[1600.,300.],converted])
    guide,_,connected=state.update(p,np.zeros(2),world,1,600.,70.,
        lambda p,a:(p.copy(),a.copy(),True,'GOAL_CONNECTED'),lambda p:p)
    assert connected
    np.testing.assert_array_equal(guide[-1],world)


def test_new_blocker_invalidates_a_previously_certified_curved_prediction():
    path=np.array([[650.,350.],[700.,250.],[820.,250.],[895.,350.]])
    obs=np.array([[760.,350.,17.],[790.,250.,17.]])
    _,_,valid=safe_goal_path(path,np.zeros(4),GOAL,obs,HULL,10.,BOUNDS,50.,
        visible_radius=400.,prediction_clearance=.4)
    assert not valid


def test_invalid_prediction_never_retains_a_previous_guide():
    state=TerminalDisplayState();update(state)
    p,_,valid=update(state,2,False,np.array([[np.nan,350.],[950.,350.]]))
    assert len(p)==0 and not valid and state.reason=='INVALID_PREDICTION'
