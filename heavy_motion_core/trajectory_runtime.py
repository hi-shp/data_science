"""The same perception-limited trajectory controller for GUI and experiments."""
import numpy as np

from heavy_motion_core.navigation_map import NavigationMap
from heavy_motion_core.perception import lidar_hits_np, update_grid
from heavy_motion_core.experiments.sampling_navigation import SamplingConfig, SamplingNavigator
from heavy_motion_core.trajectory_modes import config_for_mode


DEFAULT_TRAJECTORY_CONFIG = SamplingConfig(
    smooth_weight=3., yaw_command_step=.1, symmetric_yaw_bias=0., objective='eta')


def advance_trajectory(env, navigator=None, step_idx=0, sub_steps=1):
    """Advance every fixed physics step; plan only on the existing control cadence."""
    env.frame += 1
    env.update_dynamic_obstacles()
    scale = env.dynamics.pixels_per_m
    dists, hx, hy = lidar_hits_np(
        env.boat_pos, env.boat_heading, env.rel_angles,
        env.dynamic_obstacles, env.lidar_range,
        (0, 0, env.map_w, env.sim_h))
    env.lidar_dists = dists
    update_grid(env.grid, hx, hy)
    env.grid *= .945
    if (env.frame-1) % env.control.planning_period_steps == 0 or getattr(env, '_line_resume_plan', False):
        if not hasattr(env, 'navigation_map'):
            env.navigation_map = NavigationMap(env.map_w/scale, env.sim_h/scale)
        env.navigation_map.observe(env.boat_pos/scale, env.boat_heading,
                                   env.rel_angles, dists/scale,
                                   env.lidar_range/scale, env.frame*env.dt)
        env.perceived_obstacles = env.navigation_map.obstacles*scale
        if navigator is None:
            if not hasattr(env, 'trajectory_navigator'):
                mode = getattr(env, 'navigation_mode', 'eta_base')
                config = (SamplingConfig(yaw_command_step=.1) if mode == 'trajectory_reference'
                          else config_for_mode(mode) if mode.startswith('eta_')
                          else DEFAULT_TRAJECTORY_CONFIG)
                env.trajectory_navigator = SamplingNavigator(
                    env.dynamics, env.dt, 'trajectory_control', config)
            navigator = env.trajectory_navigator
        command, prediction, display_point, clearance = navigator.plan(
            env.physics_state(), env.navigation_map, env.target/scale, env.frame)
        env.motion_prediction_states = prediction
        env.command_speed, env.command_yaw_rate = map(float, command)
        env.raw_route = navigator.path
        env.control_path = np.vstack([env.physics_state()[:2], prediction[:, :2]])
        env.predicted_trajectory = env.control_path
        env._line_resume_plan = False
        env.prediction_frame = env.frame
        env.prediction_stride_steps = env.control.planning_period_steps
        env.controller_target = prediction[min(8, len(prediction)-1), :2]*scale
        env.pursuit_target = None if navigator.mode == 'trajectory_control' else display_point*scale
        env.heading_target = (float(prediction[min(8, len(prediction)-1), 2]) if
                              navigator.mode == 'trajectory_control' else
                              float(np.arctan2(display_point[1]-env.boat_pos[1]/scale,
                                               display_point[0]-env.boat_pos[0]/scale)))
        env.predicted_clearance = clearance
        env.min_wide_dist = float(dists.min())
        env.selected_gap = None
        env.current_wp = None
        env.next_wp = None
        env.all_gaps = []
        env.total_gaps_count = 0
    env.prev_steer = env.command_yaw_rate/env.dynamics.max_yaw_rate_rad_s
    env.step(*env.get_pwm(env.prev_steer), sub_step_idx=step_idx,
             total_sub_steps=sub_steps)
    env.update_camera()
    return hx, hy
