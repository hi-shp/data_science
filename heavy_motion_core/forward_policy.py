"""Forward feasibility for the opt-in exact-polygon controller mode.

The policy never changes the hull gate or infers free space from the world.
All thresholds here define meaningful progress/short continuation, not a
replacement for the 0.20 m physical safety gate.
"""
import numpy as np


def forward_probes(state, passage, physics, horizon, previous, yaw_step):
    """A small deterministic motion library, independent of map/seed IDs."""
    cruise = physics.cruise_speed_m_s
    vmax = physics.max_yaw_rate_rad_s
    u,r,fl,fr = state[3],state[5],state[6],state[7]
    surge_drag = physics.surge_linear_drag*u+physics.surge_quadratic_drag*u*abs(u)
    thrust_speed = np.clip(u+physics.speed_response_s/physics.mass_kg*
                           (fl+fr-surge_drag), 0., cruise)
    yaw_gain = (physics.yaw_inertia_kg_m2/physics.yaw_response_s if
                physics.yaw_rate_gain_Nm_s is None else physics.yaw_rate_gain_Nm_s)
    yaw_drag = physics.yaw_linear_drag*r+physics.yaw_quadratic_drag*r*abs(r)
    thrust_yaw = np.clip(r+((fr-fl)*physics.thruster_arm_m-yaw_drag)/yaw_gain,
                          -vmax,vmax)
    if passage is None:
        desired = 0.
    else:
        delta = passage.center-state[:2]
        entry = np.arctan2(delta[1], delta[0])
        tangent = np.arctan2(passage.tangent[1], passage.tangent[0])
        desired = np.arctan2(np.sin(.5*entry+.5*tangent-state[2]),
                             np.cos(.5*entry+.5*tangent-state[2]))
    steer = np.clip(desired / physics.yaw_response_s-.4*state[5], -vmax, vmax)
    # Straight, hold-thrust, and both turn directions with delayed onset,
    # counter-steer and release. These are feasibility probes, not ETA tuning.
    profiles = [(thrust_speed, thrust_yaw, 0, horizon),
                (cruise, 0., 0, horizon)]
    for speed in (cruise, .65*cruise):
        for direction in (-1., 1.):
            for rate in (.25*vmax, .5*vmax):
                for onset, hold in ((0, horizon//3), (horizon//5, horizon//2)):
                    profiles.append((speed, direction*rate, onset, hold))
    # Two geometry-led align/enter hypotheses use the observed passage angle.
    if passage is not None:
        profiles.extend(((cruise, steer, 0, horizon//3),
                         (.65*cruise, steer, horizon//8, horizon//2)))
    sequences = np.empty((len(profiles), horizon, 2), dtype=float)
    for i, (speed, yaw, onset, hold) in enumerate(profiles):
        sequences[i, :, 0] = speed
        sequences[i, :, 1] = 0.
        sequences[i, onset:onset+hold, 1] = yaw
        if yaw and onset == 0 and i % 2:
            sequences[i, onset+hold:onset+hold+max(2, horizon//8), 1] = -.5*yaw
    if passage is not None:
        # Align, counter-steer, then release into the exit corridor. The
        # steering sign follows the currently observed gap, never a map ID.
        entry_error = np.arctan2(np.sin(entry-state[2]),
                                  np.cos(entry-state[2]))
        sign = 1. if entry_error >= 0. else -1.
        adaptive_speed = max(.45*cruise,
                             min(max(0., state[3])+.1, .75*cruise))
        patterns = ((.3,-.3,6,20), (.2,-.3,10,15),
                    (.3,-.4,6,15), (.4,-.3,6,15),
                    (.2,-.4,10,15), (.3,-.2,6,15),
                    (.3,-.3,8,18), (.4,-.4,6,15),
                    (.2,-.3,8,20), (.4,-.3,8,18))
        extra = np.empty((len(patterns)*2, horizon, 2))
        for i, (first, second, change1, change2) in enumerate(patterns):
            for j, speed in enumerate((adaptive_speed, .65*cruise)):
                candidate = extra[2*i+j]
                candidate[:, 0] = speed
                a = min(horizon, max(1, round(change1*horizon/40)))
                b = min(horizon, max(a+1, round(change2*horizon/40)))
                candidate[:a, 1] = sign*first
                candidate[a:b, 1] = sign*second
                candidate[b:, 1] = sign*.2
        sequences = np.concatenate((sequences,extra),axis=0)
    last = np.full(len(sequences), previous[1])
    for t in range(horizon):
        sequences[:, t, 1] = np.clip(sequences[:, t, 1],
                                    last-yaw_step, last+yaw_step)
        last = sequences[:, t, 1]
    return sequences


def forward_progress(states, sequences, initial, goal, passage, path, arc,
                     knot_dt, start_s, route_projector=None):
    """Filter actual forward motion, not a positive command alone."""
    from heavy_motion_core.experiments.sampling_navigation import route_coordinates
    if route_projector is None:
        route_projector = route_coordinates
    displacement = states[:, -1, :2]-initial[:2]
    bow = np.array([np.cos(initial[2]), np.sin(initial[2])])
    body_distance = displacement@bow
    reverse_distance = np.maximum(-states[:, :, 3], 0.).sum(axis=1) * knot_dt
    goal_progress = (np.linalg.norm(goal-initial[:2])-
                     np.linalg.norm(goal-states[:, -1, :2], axis=1))
    motion_possible = ((body_distance > .35) & (states[:, -1, 3] > .15) &
                       np.all(sequences[:, :, 0] >= 0., axis=1) &
                       (reverse_distance < .12))
    route_gain = np.full(len(states), -np.inf)
    route_heading = np.full(len(states), initial[2])
    possible_indices = np.flatnonzero(motion_possible)
    if len(possible_indices):
        end_s, _, heading = route_projector(
            states[possible_indices, -1, :2], path, arc)
        route_gain[possible_indices] = end_s-start_s
        route_heading[possible_indices] = heading
    if passage is None:
        passage_gain = np.full(len(states), -np.inf)
    else:
        passage_gain = displacement@passage.tangent
    positive = (motion_possible &
                ((goal_progress > .3) | (route_gain > .3) |
                 (passage_gain > .6)))
    return (positive, np.maximum(goal_progress, route_gain), passage_gain,
            route_gain, route_heading)


def continuation_safe(navigator, terminal, last_commands, observation,
                      margin, knot_count=10, return_witness=False,
                      require_forward=True):
    """Physical short continuation, including current yaw and thrust lag.

    This checks observed obstacles only. A finite library cannot prove that a
    corridor is blocked, so a false result means only no verified continuation.
    """
    if not len(terminal):
        return np.empty(0, dtype=bool)
    physics = navigator.p
    speeds = np.clip(terminal[:, 3], .25*physics.cruise_speed_m_s,
                     physics.cruise_speed_m_s)
    yaw = np.clip(terminal[:, 5], -physics.max_yaw_rate_rad_s,
                  physics.max_yaw_rate_rad_s)
    variants = np.empty((len(terminal)*3, knot_count, 2))
    for j, adjustment in enumerate((0., -.15, .15)):
        variants[j::3, :, 0] = speeds[:, None]
        variants[j::3, :, 1] = np.clip(yaw[:, None]+adjustment,
                                        -physics.max_yaw_rate_rad_s,
                                        physics.max_yaw_rate_rad_s)
    previous = np.repeat(last_commands[:, 1], 3)
    step = navigator.cfg.yaw_command_step
    if step is not None:
        for t in range(knot_count):
            variants[:, t, 1] = np.clip(variants[:, t, 1],
                                        previous-step, previous+step)
            previous = variants[:, t, 1]
    from heavy_motion_core.experiments.fast_rollout import compiled_rollout
    initial_states = np.repeat(terminal, 3, axis=0)
    if compiled_rollout is not None:
        future, clearance = compiled_rollout(
            terminal[0], variants, observation.obstacles,
            observation.width, observation.height, navigator.dt,
            navigator.fast_params, 0., 0., 0., navigator.hull_polygons,
            None, navigator.hull_edges, navigator.hull_bound,
            navigator.hull_box, initial_states, navigator.cfg.forward_policy)
    else:
        results=[]
        for i,state in enumerate(terminal):
            results.append(navigator.rollout(state,variants[3*i:3*i+3],
                observation.obstacles,observation.width,observation.height))
        future=np.concatenate([result[0] for result in results])
        clearance=np.concatenate([result[1] for result in results])
    displacement = future[:, -1, :2]-initial_states[:, :2]
    heading = np.column_stack((np.cos(initial_states[:, 2]),
                               np.sin(initial_states[:, 2])))
    advancing = ((np.sum(displacement*heading, axis=1) > .1) &
                 (future[:, -1, 3] > .1))
    threshold=margin
    if not require_forward:
        advancing=np.ones(len(future),dtype=bool)
    safe=((clearance >= threshold) & advancing).reshape(-1,3)
    feasible=np.any(safe,axis=1)
    if return_witness:
        witness=variants.reshape(len(terminal),3,knot_count,2)[
            np.arange(len(terminal)),np.argmax(safe,axis=1)]
        return feasible,witness
    return feasible


def passage_crossed(states, initial, passage, hull_polygons):
    """Observed opening crossed and physical stern clears its plane."""
    if passage is None:
        return np.zeros(len(states), dtype=bool)
    points = np.concatenate((np.broadcast_to(initial[:2],
                             (len(states),1,2)),states[:,:,:2]),axis=1)
    rel = points-passage.center
    along = rel@passage.tangent
    across = rel[:,:,0]*passage.tangent[1]-rel[:,:,1]*passage.tangent[0]
    delta = along[:,1:]-along[:,:-1]
    fraction = np.clip(-along[:,:-1]/np.maximum(delta,1e-9),0.,1.)
    lateral = across[:,:-1]+fraction*(across[:,1:]-across[:,:-1])
    crossed = np.any((along[:,:-1]<=0.)&(along[:,1:]>=0.)&
                     (np.abs(lateral)<passage.free_width/2),axis=1)
    terminal = states[:,-1]
    vertices = hull_polygons.reshape(-1,2)
    c,s = np.cos(terminal[:,2]),np.sin(terminal[:,2])
    projected = ((c[:,None]*vertices[:,0]-s[:,None]*vertices[:,1])*passage.tangent[0]+
                 (s[:,None]*vertices[:,0]+c[:,None]*vertices[:,1])*passage.tangent[1])
    stern = (terminal[:,:2]-passage.center)@passage.tangent+projected.min(axis=1)
    return crossed & (stern>0.)


def topology_allows_forward(route_gain, direct_open, passage_gain, passage):
    """Observed A* topology vetoes goalward motion into a closed corridor."""
    return ((route_gain > .3) | direct_open |
            ((passage is not None) & (passage_gain > .6)))


def choose_forward(navigator, initial, observation, goal, path, arc, passage,
                   sequences, states, closest, families, quality, selected,
                   direct_open, prior_commitment, route_start=None):
    """Replace a reverse/stalled selection only with verified forward motion.

    The original candidate population and exact safety results remain intact.
    Extra physical probes are evaluated only at a decision boundary.
    """
    cfg = navigator.cfg
    knot_dt = 3*navigator.dt
    from heavy_motion_core.experiments.sampling_navigation import route_coordinates
    selected_end=states[selected,-1]
    motion=selected_end[:2]-initial[:2]
    selected_valid=bool(
        closest[selected]>=cfg.margin and
        np.all(sequences[selected,:,0]>=0.) and
        selected_end[3]>.15 and
        motion@np.array([np.cos(initial[2]),np.sin(initial[2])])>.35 and
        np.maximum(-states[selected,:,3],0.).sum()*knot_dt<.12)
    prior_passage = (prior_commitment is not None and passage is not None and
                     np.linalg.norm(prior_commitment-passage.center)<.8)
    selected_crossed = (passage_crossed(states[selected:selected+1],initial,
                        passage,navigator.hull_polygons)[0] if prior_passage
                        else False)
    trigger = (not selected_valid or
               (prior_passage and not selected_crossed))
    if not trigger:
        if prior_passage and selected_crossed:
            families[selected]='PASSAGE'
        return selected,sequences,states,closest,families,dict(
            activated=False,probes=0,viable=1,
            selected_source='existing',selected_forward=True)

    if route_start is None:
        route_start = route_coordinates(initial[:2],path,arc)[0][0]
    movement, progress, passage_gain, route_gain, _ = forward_progress(
        states,sequences,initial,goal,passage,path,arc,knot_dt,route_start)
    # Unknown/free-space topology is inherited from the observed A* route.
    # A goalward detour is allowed when the local direct corridor or a measured
    # passage supports it; otherwise it must also advance along the route.
    topological = topology_allows_forward(route_gain,direct_open,
                                           passage_gain,passage)
    valid = movement & topological & (closest >= cfg.margin)
    crossed = passage_crossed(states,initial,passage,navigator.hull_polygons)
    online_viable=np.zeros(len(sequences),dtype=bool)
    online_indices=np.flatnonzero(valid)
    if len(online_indices):
        online_viable[online_indices]=continuation_safe(navigator,
            states[online_indices,-1],sequences[online_indices,-1],
            observation,cfg.margin)
    need_probes=(not np.any(online_viable) or
                 (prior_passage and not np.any(online_viable&crossed)))
    if need_probes:
        probes = forward_probes(initial,passage,navigator.p,cfg.horizon_steps,
                                navigator.previous,cfg.yaw_command_step)
        probe_states,probe_clearance = navigator.rollout(
            initial,probes,observation.obstacles,
            observation.width,observation.height,goal)
        extra_move,extra_progress,extra_gap,extra_route,_ = forward_progress(
            probe_states,probes,initial,goal,passage,path,arc,knot_dt,route_start)
        extra_topology = topology_allows_forward(extra_route,direct_open,
                                                 extra_gap,passage)
        extra_valid = extra_move & extra_topology & (probe_clearance>=cfg.margin)
        extra_crossed=passage_crossed(probe_states,initial,passage,
                                      navigator.hull_polygons)
        extra_viable=np.zeros(len(probes),dtype=bool)
        extra_indices=np.flatnonzero(extra_valid)
        if len(extra_indices):
            extra_viable[extra_indices]=continuation_safe(navigator,
                probe_states[extra_indices,-1],probes[extra_indices,-1],
                observation,cfg.margin)
        all_sequences=np.concatenate((sequences,probes))
        all_states=np.concatenate((states,probe_states))
        all_clearance=np.concatenate((closest,probe_clearance))
        all_crossed=np.concatenate((crossed,extra_crossed))
        all_progress=np.concatenate((progress,extra_progress))
        all_gap=np.concatenate((passage_gain,extra_gap))
        all_families=np.concatenate((families,np.full(len(probes),'CURRENT',dtype='<U16')))
        viable=np.concatenate((online_viable,extra_viable))
    else:
        probes=np.empty((0,cfg.horizon_steps,2))
        all_sequences,all_states,all_clearance,all_crossed=sequences,states,closest,crossed
        all_progress,all_gap,all_families,viable=progress,passage_gain,families,online_viable
    eligible=np.flatnonzero(viable)
    if not len(eligible):
        return selected,all_sequences,all_states,all_clearance,all_families,dict(
            activated=True,probes=len(probes),viable=0,
            selected_source='existing_recovery',selected_forward=False)
    if prior_passage and np.any(all_crossed[eligible]):
        eligible=eligible[all_crossed[eligible]]
    # A continuing strategy or warm start wins among similarly advancing
    # motions. Otherwise choose actual forward progress and then smoothness.
    best_progress=np.max(all_progress[eligible])
    near=eligible[all_progress[eligible]>=best_progress-.35]
    prior_family=navigator.strategy
    if prior_family is not None and prior_family!='RECOVERY':
        matching=near[all_families[near]==prior_family]
        if len(matching):near=matching
    if 0 in near:chosen=0
    else:
        if len(near)>1:
            best_gap=np.max(all_gap[near])
            if passage is not None:
                near=near[all_gap[near]>=best_gap-.35]
        q=np.empty(len(near))
        for j,i in enumerate(near):
            if i<len(quality):q[j]=quality[i]
            else:
                u=all_sequences[i]
                delta=np.diff(np.vstack((navigator.previous,u)),axis=0)
                q[j]=np.sum(delta[:,1]**2)+.2*np.sum(delta[:,0]**2)
        chosen=int(near[np.argmin(q)])
    if passage is not None and (all_crossed[chosen] or
                                (prior_passage and all_gap[chosen]>.6)):
        all_families[chosen]='PASSAGE'
    return chosen,all_sequences,all_states,all_clearance,all_families,dict(
        activated=True,probes=len(probes),viable=int(viable.sum()),
        viable_original=int(viable[:len(sequences)].sum()),
        viable_probes=int(viable[len(sequences):].sum()),
        selected_source='probe' if chosen>=len(sequences) else 'existing',
        selected_forward=True,selected_passage=bool(all_crossed[chosen]))
