"""V2.1: pinned CODEX control, with downstream GAP/Bezier display annotations.

Neither a GAP annotation nor the pursuit reference is a controller input.
V1 remains available through MAIN_HEAVY_MOTION_VERSION=V1.
"""
import os
import json
import math
import time
from pathlib import Path

import numpy as np

from heavy_motion_core.controller_config import ControllerParameters
from heavy_motion_core.trajectory_runtime import advance_trajectory
from heavy_motion_core.control_path import path_geometry
from heavy_motion_core.trajectory_display import future_trajectory
from heavy_motion_core.passage_geometry import physical_hull_polygons
from heavy_gap_annotation import (route_bezier, route_crossings,
                                  select_route_gaps, clipped_display_route,
                                  gap_identity,crossing_in_front,forward_crossings)
from heavy_gap_diagnostics import compute_legacy_gap_metrics
from heavy_gap_profile import load_presentation_profile
from heavy_gap_state import GapAnnotationState
from utils import pure_pursuit
from perception import update_grid, match_clusters
from heavy_gap_display import MainDisplayClusters
from pursuit_marker_display import PursuitDisplayMarker


class HeavyMotionV2:
    version = 'V2.1_GAP_PERSISTENCE'

    def __init__(self, env):
        env.motion_core_v2 = True
        env.navigation_mode = 'eta_continuity_forward'
        data = json.loads(Path(__file__).with_name('vessel_config.json').read_text())
        env.control = ControllerParameters(**data['controller'])
        env.phase5_visuals = self
        env.show_all_gaps = True
        self.presentation_profile = load_presentation_profile(env.dynamics.pixels_per_m)
        self.annotation_state = GapAnnotationState(physical_hull_polygons(env.dynamics.pixels_per_m)*env.dynamics.pixels_per_m)
        self.worker = None
        self.pursuit_marker = PursuitDisplayMarker()
        self.reset_episode(env)

    def close(self):
        if self.worker is not None:
            self.worker.close()
            self.worker = None

    def start(self, env):
        if os.environ.get('MAIN_HEAVY_SYNC_MOTION','') != '1':
            from heavy_motion_worker import MotionCommandWorker
            self.worker = MotionCommandWorker(env)
            self.worker.prime()

    def available(self):
        return None if self.worker is None else self.worker.available()

    def invalidate_controller(self, env):
        self.reset_episode(env, controller_only=True)

    def reset_episode(self, env, *, controller_only=False):
        self.annotation_state.reset()
        self.pursuit_marker.reset()
        env.visual_pursuit_target = None
        for key in ('navigation_map', 'trajectory_navigator', 'motion_prediction_states'):
            if hasattr(env, key):
                delattr(env, key)
        self.last_frame = -1
        self.last_generation = -1
        self.last_result = None
        self.reference = None
        self.cached_segments = []
        self.gui_grid = np.zeros_like(env.grid) if hasattr(env, 'grid') else None
        self.gui_clusters, self.gui_ids = [], []
        self.gui_cluster_cache = MainDisplayClusters()
        self.gui_lidar_dists = None
        self.gui_hits = (None, None)
        env.raw_route = env.control_path = env.predicted_trajectory = None
        env.controller_target = None
        env.command_speed = env.command_yaw_rate = 0.
        env.current_wp = env.next_wp = None
        env.candidate_wps = []
        env.clusters, env.cluster_ids = [], []
        env.all_gaps = []
        env.total_gaps_count = 0
        env.bezier_path = env.next_bezier_path = None
        env.pursuit_target = env.next_pursuit_target = None
        env.visual_selected_trajectory = None
        env.visual_portal_segments = []
        env.prediction_frame = 0
        env.predicted_clearance = math.inf
        env.perceived_obstacles = np.empty((0, 3))
        env.selected_gap = None
        if not controller_only:
            env.wakes = []
            env.reflected_wakes = []
        if self.worker is not None and (not controller_only or not env.linetrace_mode):
            self.worker.reset(env)

    def advance(self, env, step_idx=0, sub_steps=1):
        if env.frame < self.last_frame:
            self.reset_episode(env)
        if self.worker is None:
            hits = advance_trajectory(env, step_idx=step_idx, sub_steps=sub_steps)
        else:
            packet = self.worker.take()
            if packet['frame'] != env.frame+1:
                raise RuntimeError('Completed command belongs to a different physics tick')
            env.frame += 1
            env.update_dynamic_obstacles()
            prediction = packet.get('prediction')
            if prediction is not None:
                env.__dict__.update(prediction)
            for key in ('lidar_dists','prev_steer','command_speed','command_yaw_rate',
                        'heading_target','min_wide_dist','prediction_frame'):
                setattr(env,key,packet[key])
            env.step(*packet['pwm'],sub_step_idx=step_idx,total_sub_steps=sub_steps)
            if not np.array_equal(env.physics_state(),packet['state']):
                raise RuntimeError('Completed CODEX command diverged from authoritative physics')
            env.update_camera()
            hits = packet['hits']
        self.last_frame = env.frame
        self.update_gui_scan(env, hits)
        # Display invalidation is independent of prediction/control cadence.
        # Never keep a cached waypoint behind the current bow between plans.
        behind=any(gap is not None and not crossing_in_front(gap,env.boat_pos,env.boat_heading)
                   for gap in (env.current_wp,env.next_wp))
        completed=self.annotation_state.completion_reason(env.current_wp,env.boat_pos,env.boat_heading)
        if env.prediction_frame != self.last_generation or behind or completed is not None:
            self.last_generation = env.prediction_frame
            self._annotate(env)
        return hits

    def update_gui_scan(self, env, hits):
        """MAIN sees buoy rays only; the CODEX sensor/control scan stays intact."""
        hx, hy = hits
        epsilon = 4*np.finfo(np.float32).eps*max(env.map_w,env.sim_h,env.lidar_range)
        wall = np.isfinite(hx) & ((np.abs(hx)<epsilon) |
            (np.abs(hx-env.map_w)<epsilon) | (np.abs(hy)<epsilon) |
            (np.abs(hy-env.sim_h)<epsilon))
        self.gui_hits = (np.where(wall, np.nan, hx), np.where(wall, np.nan, hy))
        self.gui_lidar_dists = np.where(wall, env.lidar_range, env.lidar_dists)
        if self.gui_grid is not None:
            update_grid(self.gui_grid, *self.gui_hits)
            self.gui_grid *= .945

    def render(self, env, fallback_hits):
        """Display-only sensor view; restore the authoritative scan immediately."""
        if env.manual_mode or env.linetrace_mode:
            env.visual_pursuit_target = env.pursuit_target
            return env.renderer.render(*fallback_hits)
        if self.gui_lidar_dists is None:
            self.update_gui_scan(env, fallback_hits)
        original = env.lidar_dists
        original_grid = env.grid
        try:
            marker = self.pursuit_marker.update(
                time.perf_counter(), float(np.linalg.norm(getattr(env, 'boat_vel', (0., 0.)))),
                getattr(env, 'dt', .04),
                paused=getattr(env, 'paused', False))
            env.visual_pursuit_target = env.pursuit_target if marker is None else marker
            if self.pursuit_marker.at_stop:
                self.annotation_state.latch_first(env.current_wp)
            env.lidar_dists = self.gui_lidar_dists
            env.grid = self.gui_grid
            return env.renderer.render(*self.gui_hits)
        finally:
            env.lidar_dists = original
            env.grid = original_grid

    def _annotate(self, env):
        """MAIN buoy-only clustering/candidates, with route crossing annotations."""
        scale = env.dynamics.pixels_per_m
        obs = env.navigation_map.obstacles*scale
        if self.gui_grid is not None:
            self.gui_clusters, self.gui_ids = match_clusters(self.gui_clusters,
                self.gui_ids, self.gui_cluster_cache.extract(self.gui_grid))
        env.clusters, env.cluster_ids = self.gui_clusters, self.gui_ids
        heading = float(env.boat_heading) if hasattr(env, 'boat_heading') else float(env.physics_state()[2])
        ch, sh = math.cos(heading), math.sin(heading)
        # Same MAIN GAPS toggle population and midpoint candidate glyphs.
        front = [(c, i) for c, i in zip(self.gui_clusters, self.gui_ids)
                 if (c[0]-env.boat_pos[0])*ch+(c[1]-env.boat_pos[1])*sh >= 0.]
        candidates = [dict(c1=c1.copy(), c2=c2.copy(), pos=(c1+c2)/2., pair=(i1,i2))
                      for index,(c1,i1) in enumerate(front) for c2,i2 in front[index+1:]]
        env.total_gaps_count = len(front)*(len(front)-1)//2
        env.all_gaps = candidates if env.show_all_gaps else []
        # Motion has already been selected by CODEX. Only that prediction
        # determines the display route and its GAP order; no MAIN score calls.
        original = np.asarray(env.control_path)*scale
        path = future_trajectory(original, env.boat_pos,
            env.frame-env.prediction_frame, getattr(env,'prediction_stride_steps',3))
        keep = np.r_[True,np.linalg.norm(np.diff(path,axis=0),axis=1)>1e-8]
        path = path[keep]
        states = getattr(env,'motion_prediction_states',None)
        if states is not None and len(states)+1==len(original):
            original_heading = np.unwrap(np.r_[heading,np.asarray(states)[:,2]])
            segments, lengths, arc = path_geometry(original)
            fraction = np.clip(np.sum((path[:,None]-original[None,:-1])*segments[None],axis=2)/
                               np.maximum(lengths[None]**2,1e-12),0.,1.)
            projected = original[None,:-1]+fraction[:,:,None]*segments[None]
            index = np.argmin(np.sum((projected-path[:,None])**2,axis=2),axis=1)
            progress = arc[index]+fraction[np.arange(len(path)),index]*lengths[index]
            path_headings = np.interp(progress,arc,original_heading)
        elif len(path)>1:
            delta = np.diff(path,axis=0)
            path_headings = np.unwrap(np.r_[np.arctan2(delta[:,1],delta[:,0]),
                                            math.atan2(delta[-1,1],delta[-1,0])])
        else:
            path_headings = np.array([heading])
        display, display_headings = route_bezier(path,path_headings)
        # Annotation may use ONLY the very same existing buoy-pair candidates
        # as the MAIN GAPS display, even when its visibility toggle is off.
        # Do not create a second all-around population just for waypoints.
        crossings=route_crossings(display,display_headings,candidates,obs,
            physical_hull_polygons(scale)*scale,env.control.safety_margin_m*scale,
            (env.map_w,env.sim_h),scale)
        crossings=forward_crossings(crossings,env.boat_pos,heading)
        candidate_by_pair={gap_identity(g):g for g in candidates}
        hull=physical_hull_polygons(scale)*scale
        speed=float(np.linalg.norm(env.boat_vel)) if hasattr(env,'boat_vel') else 0.
        response=env.dynamics.actuator_tau_s+env.control.planning_period_steps*getattr(env,'dt',.04)
        margin=env.control.safety_margin_m*scale
        def validate_pinned(gap):
            # Persistence holds the pair identity, not a vanished/old segment.
            # Current endpoints must belong to the current GUI candidate set.
            candidate=candidate_by_pair.get(gap_identity(gap))
            if candidate is None:
                return None,'candidate_absent'
            valid=route_crossings(display,display_headings,[candidate],obs,hull,
                                  margin,(env.map_w,env.sim_h),scale)
            if not valid:
                return None,'no_safe_route_crossing'
            valid=forward_crossings(valid,env.boat_pos,heading)
            if not valid:
                return None,'behind_bow'
            return min(valid[0].get('presentation_candidates',(valid[0],)),
                       key=lambda g:g['route_arc']),'valid'
        selected,following=self.annotation_state.update(crossings,display,display_headings,
            hull,speed,response,margin,self.presentation_profile,env.boat_pos,env.frame,validate_pinned,heading=heading,generation=env.prediction_frame)
        # MAIN marks the completed pair visited in both orders, for this episode.
        # This is annotation bookkeeping only; the motion core never reads it.
        if hasattr(env,'visited'):
            for pair in self.annotation_state.completed_identities:
                env.visited.add(pair);env.visited.add((pair[1],pair[0]))
        env.bezier_path,env.next_bezier_path,selected,following=clipped_display_route(
            display,selected,following,np.asarray(env.target),1.4*scale)
        self.annotation_state.apply_visible(selected,following,env.frame)
        # Selection is complete before any legacy diagnostics are computed.
        for gap in (selected,following):
            if gap is not None:
                gap.update(compute_legacy_gap_metrics(gap,env.boat_pos,heading,
                    env.target,obs,getattr(env,'params',None)))
        env.current_wp,env.next_wp = selected,following
        # A presentation alternative is still the same passage: do not show
        # its old canonical representative as an extra selected candidate.
        env.candidate_wps = [g for g in crossings if not any(
            gap_identity(member)==gap_identity(chosen)
            for member in g.get('presentation_candidates',(g,))
            for chosen in (selected,following) if chosen is not None)][:2]
        current_reference=(env.bezier_path if env.next_bezier_path is None else
            np.vstack((env.bezier_path,env.next_bezier_path[1:])))
        env.pursuit_target=pure_pursuit(current_reference,env.boat_pos,lookahead=70)
        env.next_pursuit_target=None
        self.pursuit_marker.set_path(current_reference, env.pursuit_target,
            env.prediction_frame, speed,
            env.control.planning_period_steps*getattr(env, 'dt', .04), scale,
            stop=None if selected is None else selected['pos'])
        if not self.pursuit_marker.path_continuous:
            self.annotation_state.first_latched=False
        elif self.annotation_state.first_latched:
            self.pursuit_marker.hold_at_stop()
        self.reference = current_reference
        self.last_result = dict(portal=selected,
            crossing_s=None if selected is None else selected['portal_s'],
            prediction=path, bezier_reference=self.reference,
            safe=bool(env.predicted_clearance >= env.control.safety_margin_m),
            family='CODEX_FORWARD', generation=env.prediction_frame,
            annotation=self.annotation_state.diagnostics(env.frame))
        env.visual_selected_trajectory = None
        env.visual_portal_segments = []

    def gap_interval(self, env, gap):
        return None if gap is None else gap.get('interval')

    def prepare_display(self, env):
        """Compatibility hook: rendering only draws the planning-tick cache."""
        pass



def warmup(env):
    """Match CODEX kernel warmup before the wall-clock scheduler starts."""
    from heavy_motion_core.fast_astar import warmup_astar
    from heavy_motion_core.fast_clearance import warmup_clearance
    from heavy_motion_core.fast_corridor import compiled_within_corridor
    from heavy_motion_core.experiments.fast_rollout import compiled_rollout, parameter_vector
    from heavy_motion_core.passage_geometry import (
        physical_hull_polygons, prepare_hull_edges, fast_surface_clearances)
    from heavy_gap_display import warmup as warmup_display
    warmup_display()
    warmup_astar()
    warmup_clearance()
    if compiled_within_corridor is not None:
        compiled_within_corridor(np.zeros((1, 2)), np.zeros((1, 2)),
                                 np.zeros((1, 2)), np.ones(1))
    if compiled_rollout is not None:
        scale = env.dynamics.pixels_per_m
        hull = physical_hull_polygons(scale)
        edges, bound, box = prepare_hull_edges(hull)
        args = (np.zeros(8), np.zeros((1, 1, 2)), np.empty((0, 3)),
                env.map_w/scale, env.sim_h/scale, env.dt, parameter_vector(env.dynamics))
        compiled_rollout(*args, 0., 0., 1.4, hull, None, edges, bound, box)
        fast_surface_clearances(np.zeros((1, 8)), np.empty((0, 3)),
                                env.map_w/scale, env.sim_h/scale, edges, bound, box)
        # Precompile the UI annotation's exact-wall crossing check as well.
        fast_surface_clearances(np.zeros((1, 3)), np.empty((0, 3)),
                                env.map_w/scale, env.sim_h/scale, edges, bound, box, True)
        compiled_rollout(*args, np.float32(0.), np.float32(0.), 0., hull, None, edges, bound, box,
                         np.zeros((1, 8)), True)

        compiled_rollout(*args, np.float32(0.), np.float32(0.), 1.4, hull,
                         None, edges, bound, box, None, True)
        compiled_rollout(*args, 0., 0., 0., hull, None, edges, bound, box,
                         np.zeros((1, 8)), True)
