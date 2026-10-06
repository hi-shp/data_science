"""Fast, deterministic evaluation through MAIN's actual navigation loop.

Only rendering, screenshot output, and wall-clock pacing are replaced. Every
physics/control/planning step still comes from main.run().
"""
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
from time import perf_counter

os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pygame


class Finished(Exception):
    pass


class _Renderer:
    def __init__(self, env):
        self.engine_3d = None

    def render(self, hits_x, hits_y):
        pass


def run(seeds, output, timeout_s, trace_seed=None, diagnose=False,
        shadow_horizon=0, stop_on_failure=False):
    import environment
    import main
    if shadow_horizon:
        from dynamic_path_feasibility import evaluate_candidate
        from portal_navigation import path_has_hull_clearance

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
    if [row['seed'] for row in rows] != seeds[:len(rows)]:
        raise ValueError('Existing evaluation rows are not this seed prefix')
    if stop_on_failure and any(row['outcome'] != 'success' for row in rows):
        return rows
    if len(rows) == len(seeds):
        return rows

    original_renderer = environment.EnvRenderer
    original_factory = main.BoatEnv
    original_capture = main.save_episode_frame
    original_start_capture = main.start_capture_worker
    original_time = main.time
    original_guard = main.MainSafetyGuard
    original_find_gap = main.find_gap
    from momentum_gap_router import portal_crossing
    clock = {'wall': 0.0}
    state = {'seed': seeds[len(rows)], 'active': False, 'guard': None,
             'last_pos': None, 'last_heading': None, 'distance': 0.0,
             'turn_deg': 0.0, 'last_sign': 0, 'reversals': 0,
             'speed_sum': 0.0, 'steps': 0, 'guard_interventions': 0,
             'shadow_records': [], 'shadow_cpu_s': 0.0,
             'last_diff': None, 'command_tv': 0.0,
             'straight_steps': 0, 'straight_tv': 0.0,
             'straight_yaw_sq': 0.0, 'straight_heading_sq': 0.0,
             'straight_lateral_sq': 0.0, 'straight_last_diff': None,
             'straight_last_sign': 0, 'straight_reversals': 0,
             'portal_crossings': [], 'portal_switches': 0,
             'last_portal_pair': None}
    trace = []
    recent = deque(maxlen=125)

    class FakeClock:
        def tick(self, fps=120):
            clock['wall'] += .04 / main.BASE_PLAYBACK_RATE
            return 1000*.04 / main.BASE_PLAYBACK_RATE

    class FakeTime:
        @staticmethod
        def perf_counter():
            return clock['wall']

    def guard_factory():
        guard = original_guard()
        state['guard'] = guard
        return guard

    def find_gap(*args, **kwargs):
        result = original_find_gap(*args, **kwargs)
        if not kwargs.get('is_next_wp'):
            state['new_wp'] = None if result is None else [round(float(v), 1) for v in result['pos']]
        return result

    def factory():
        random.seed(state['seed'])
        np.random.seed(state['seed'])
        env = original_factory()
        env.clock = FakeClock()
        original_step, original_reset = env.step, env.reset

        def step(*args, **kwargs):
            if not state['active']:
                state['active'] = True
                state['last_pos'] = env.boat_pos.copy()
                state['last_heading'] = env.boat_heading
                state['guard_interventions'] = state['guard'].interventions
                state['guard_predictions'] = state['guard'].predictions
                state['guard_alarms'] = state['guard'].alarms
            left, right = float(args[0]), float(args[1])
            router = getattr(env, 'momentum_gap_router', None)
            active_gap = env.current_wp if router is not None else None
            active_pair = (None if active_gap is None else
                           tuple(map(int, active_gap['pair'])))
            if (active_pair is not None and state['last_portal_pair'] is not None
                    and active_pair != state['last_portal_pair']):
                state['portal_switches'] += 1
            state['last_portal_pair'] = active_pair
            before_position = np.asarray(env.boat_pos).copy()
            differential = (right-left)/800.0
            if state['last_diff'] is not None:
                state['command_tv'] += abs(differential-state['last_diff'])
            state['last_diff'] = differential
            goal_angle = math.atan2(float(env.target[1]-env.boat_pos[1]),
                                    float(env.target[0]-env.boat_pos[0]))
            heading_error = (goal_angle-env.boat_heading+math.pi)%(2*math.pi)-math.pi
            front_angles = np.abs(env.rel_angles) <= math.pi/6
            front_clear = (getattr(env, 'lidar_dists', None) is not None and
                           np.min(env.lidar_dists[front_angles]) > 150.0)
            if front_clear and abs(heading_error) < 0.15:
                state['straight_steps'] += 1
                state['straight_yaw_sq'] += env.boat_ang_vel**2
                state['straight_heading_sq'] += heading_error**2
                state['straight_lateral_sq'] += (env.boat_pos[1]-env.target[1])**2
                if state['straight_last_diff'] is not None:
                    state['straight_tv'] += abs(differential-state['straight_last_diff'])
                state['straight_last_diff'] = differential
                straight_sign = 1 if differential > 0.005 else -1 if differential < -0.005 else 0
                if straight_sign and state['straight_last_sign'] and straight_sign != state['straight_last_sign']:
                    state['straight_reversals'] += 1
                if straight_sign:
                    state['straight_last_sign'] = straight_sign
            else:
                state['straight_last_diff'] = None
                state['straight_last_sign'] = 0
            sign = 1 if right-left > 2 else -1 if left-right > 2 else 0
            if sign and state['last_sign'] and sign != state['last_sign']:
                state['reversals'] += 1
            if sign:
                state['last_sign'] = sign
            if shadow_horizon and env.frame % 3 == 0:
                started = perf_counter()
                prediction = evaluate_candidate(env, env.bezier_path,
                                                shadow_horizon,
                                                first_pwm=(args[0], args[1]))
                geometry_safe = path_has_hull_clearance(
                    env.bezier_path, env.dynamic_obstacles,
                    (env.left_hull_local, env.right_hull_local,
                     env.deck_local), margin_px=0.0)
                state['shadow_cpu_s'] += perf_counter()-started
                state['shadow_records'].append({
                    'step': env.frame, 'time': round(env.frame*env.dt, 2),
                    'collision_step': prediction['collision_step'],
                    'geometry_safe': bool(geometry_safe),
                    'cross_track_px': round(prediction['maximum_cross_track_px'], 2),
                    'saturation_steps': prediction['steering_saturation_steps'],
                    'wp_pair': None if env.current_wp is None else tuple(map(int, env.current_wp['pair'])),
                    'path_endpoint': None if env.bezier_path is None else
                        tuple(round(float(v), 1) for v in env.bezier_path[-1]),
                })
            result = original_step(*args, **kwargs)
            if active_gap is not None:
                interval = router.gap_interval(env, active_gap)
                crossed, crossing_s = portal_crossing(
                    before_position, env.boat_pos, active_gap, interval,
                    env.target)
                if crossed:
                    state['portal_crossings'].append({
                        'pair': active_pair,
                        's': round(float(crossing_s), 4),
                        'heading_rad': round(float(env.boat_heading), 4),
                        'time_sim_s': round(env.frame * env.dt, 2),
                    })
            if diagnose:
                recent.append({
                    't': round(env.frame*env.dt, 2),
                    'x': float(env.boat_pos[0]), 'y': float(env.boat_pos[1]),
                    'heading': float(env.boat_heading),
                    'yaw': float(env.boat_ang_vel),
                    'speed': float(np.linalg.norm(env.boat_vel))/env.dynamics.pixels_per_m,
                    'target_heading': float(env.heading_target),
                    'steer': float(env.prev_steer),
                    'min_wide': float(getattr(env, 'min_wide_dist', 999.)),
                    'wp_pair': None if env.current_wp is None else tuple(map(int, env.current_wp['pair'])),
                    'wp_pos': None if env.current_wp is None else tuple(map(float, env.current_wp['pos'])),
                    'next_pair': None if env.next_wp is None else tuple(map(int, env.next_wp['pair'])),
                    'guard_interventions': state['guard'].interventions,
                })
            if state['seed'] == trace_seed and (os.environ.get('MAIN_HEAVY_PORTAL_TRACE') == '1' or state['steps'] % 5 == 0):
                from main_safety_kernels import packed_hulls, preview_hull_surface_clearance
                hulls, sizes = packed_hulls((env.left_hull_local, env.right_hull_local, env.deck_local))
                actual_clearance = preview_hull_surface_clearance(
                    env.boat_pos[0], env.boat_pos[1], env.boat_heading,
                    env.dynamic_obstacles, hulls, sizes, env.map_w, env.sim_h)
                current = env.current_wp
                path = env.bezier_path
                next_path = env.next_bezier_path
                path_tangent = None
                if path is not None and len(path) >= 2:
                    tangent = path[-1] - path[-2]
                    path_tangent = round(math.atan2(float(tangent[1]), float(tangent[0])), 3)
                trace.append({
                    'left_pwm': left, 'right_pwm': right,
                    'surge': float(env.physics_state()[3]),
                    'actual_clearance_m': float(actual_clearance/env.dynamics.pixels_per_m),
                    't': round(env.frame*env.dt, 2),
                    'x': round(float(env.boat_pos[0]), 1),
                    'y': round(float(env.boat_pos[1]), 1),
                    'heading': round(float(env.boat_heading), 3),
                    'yaw': round(float(env.boat_ang_vel), 3),
                    'speed': round(float(np.linalg.norm(env.boat_vel))/env.dynamics.pixels_per_m, 3),
                    'target_heading': round(float(env.heading_target), 3),
                    'steer': round(float(env.prev_steer), 3),
                    'desired_speed': round(float(env.command_speed), 3),
                    'min_wide': round(float(getattr(env, 'min_wide_dist', 999.0)), 1),
                    'wp': None if env.current_wp is None else [round(float(v), 1) for v in env.current_wp['pos']],
                    'next_wp': None if env.next_wp is None else [round(float(v), 1) for v in env.next_wp['pos']],
                    'new_wp': state.get('new_wp'),
                    'candidate_pairs': [list(map(int, candidate['pair']))
                                        for candidate in getattr(env, 'candidate_wps', [])[:8]],
                    'pursuit': None if env.pursuit_target is None else [round(float(v), 1) for v in env.pursuit_target],
                    'anticipation_blend': round(float(getattr(env, 'anticipation_blend', 0.0)), 3),
                    'guard': state['guard'].interventions,
                    'portal_pair': None if current is None else list(map(int, current['pair'])),
                    'phase5_new_pair': getattr(env, 'phase5_new_wp_pair', None),
                    'portal_endpoints': None if current is None else [list(map(float, current['c1'])), list(map(float, current['c2']))],
                    'portal_interval': None if current is None else current.get('portal_safe_interval'),
                    'portal_s': None if current is None else current.get('portal_s'),
                    'portal_heading': None if current is None else current.get('portal_heading'),
                    'gap_candidate_count': getattr(env, 'portal_gap_count', None),
                    'portal_pair_visible': getattr(env, 'portal_pair_visible', None),
                    'portal_pair_visited': getattr(env, 'portal_pair_visited', None),
                    'portal_decisions': getattr(env, 'portal_decisions', None),
                    'portal_selection_failed': getattr(env, 'portal_initial_selection_failed', False),
                    'path_id': id(path),
                    'path_endpoint': None if path is None else list(map(float, path[-1])),
                    'path_terminal_tangent': path_tangent,
                    'next_path_id': id(next_path),
                    'next_path_endpoint': None if next_path is None else list(map(float, next_path[-1])),
                    'direct_mode': current is None,
                    'emergency_mode': bool(env.emergency_mode),
                    'portal_no_safe_route': bool(getattr(env, 'portal_no_safe_route', False)),
                    'dynamic_path_reason': getattr(getattr(env, 'dynamic_path_selector', None),
                                                   'last_reason', None),
                    'momentum_family': (getattr(getattr(env, 'momentum_gap_router', None),
                                                'last_result', None) or {}).get('family')
                    if getattr(env, 'momentum_gap_router', None) is not None else None,
                    'momentum_safe': (getattr(getattr(env, 'momentum_gap_router', None),
                                              'last_result', None) or {}).get('safe')
                    if getattr(env, 'momentum_gap_router', None) is not None else None,
                    'momentum_progress_px': (getattr(getattr(env, 'momentum_gap_router', None),
                                                     'last_result', None) or {}).get('progress_px')
                    if getattr(env, 'momentum_gap_router', None) is not None else None,
                    'momentum_detour_active': (getattr(env.momentum_gap_router,
                                                      'last_detour_active')
                                                if getattr(env, 'momentum_gap_router', None)
                                                is not None else None),
                    'momentum_observations': (getattr(env.momentum_gap_router,
                                                      'last_observations').tolist()
                                              if getattr(env, 'momentum_gap_router', None) is not None
                                              else None),
                    'momentum_scores': (getattr(env.momentum_gap_router, 'last_scores')
                                        if getattr(env, 'momentum_gap_router', None) is not None
                                        else None),
                    'momentum_candidate_states': (
                        getattr(env.momentum_gap_router, 'last_candidate_states')
                        if getattr(env, 'momentum_gap_router', None) is not None
                        else None),
                })
            state['distance'] += float(np.linalg.norm(env.boat_pos-state['last_pos']))/env.dynamics.pixels_per_m
            state['turn_deg'] += math.degrees(abs(env.boat_heading-state['last_heading']))
            state['last_pos'] = env.boat_pos.copy()
            state['last_heading'] = env.boat_heading
            state['speed_sum'] += float(np.linalg.norm(env.boat_vel))/env.dynamics.pixels_per_m
            state['steps'] += 1
            return result

        def reset():
            if state['active']:
                distance = float(np.linalg.norm(env.target-env.boat_pos))
                collision = bool(env.collide())
                outcome = ('timeout' if env.frame*env.dt >= timeout_s and
                           not collision and distance >= 70 else
                           'collision' if collision else
                           'success' if distance < 70 else None)
                if outcome is None:
                    raise RuntimeError('Unexpected episode reset')
                row = {
                    'seed': state['seed'],
                    'map_hash': hashlib.sha256(env.obstacles.tobytes()).hexdigest()[:16],
                    'outcome': outcome,
                    'completion_sim_s': round(env.frame*env.dt, 4),
                    'path_length_m': round(state['distance'], 4),
                    'average_speed_m_s': round(state['speed_sum']/state['steps'], 4),
                    'cumulative_turn_deg': round(state['turn_deg'], 2),
                    'steering_reversals': state['reversals'],
                    'command_total_variation': round(state['command_tv'], 4),
                    'straight_steps': state['straight_steps'],
                    'straight_steering_reversals': state['straight_reversals'],
                    'straight_command_tv': round(state['straight_tv'], 4),
                    'straight_yaw_rms_rad_s': round(math.sqrt(state['straight_yaw_sq']/max(1, state['straight_steps'])), 5),
                    'straight_heading_rms_rad': round(math.sqrt(state['straight_heading_sq']/max(1, state['straight_steps'])), 5),
                    'straight_lateral_rms_px': round(math.sqrt(state['straight_lateral_sq']/max(1, state['straight_steps'])), 3),
                    'safety_interventions': state['guard'].interventions-state['guard_interventions'],
                    'safety_predictions': state['guard'].predictions-state['guard_predictions'],
                    'safety_alarms': state['guard'].alarms-state['guard_alarms'],
                }
                if getattr(env, 'momentum_gap_router', None) is not None:
                    row['portal_crossings'] = state['portal_crossings']
                    row['portal_switches'] = state['portal_switches']
                if shadow_horizon:
                    detections = [item for item in state['shadow_records']
                                  if item['collision_step']]
                    first = detections[0] if detections else None
                    row['shadow'] = {
                        'horizon_steps': shadow_horizon,
                        'probe_count': len(state['shadow_records']),
                        'predicted_collision_count': len(detections),
                        'first_detection': first,
                        'first_detection_lead_steps': None if first is None else
                            env.frame-first['step'],
                        'geometry_unsafe_count': sum(not item['geometry_safe']
                                                     for item in state['shadow_records']),
                        'cpu_ms_per_probe': round(1000*state['shadow_cpu_s']/max(1, len(state['shadow_records'])), 3),
                    }
                selector = getattr(env, 'dynamic_path_selector', None)
                if selector is not None:
                    row['dynamic_path'] = {
                        'probes': selector.probes, 'rejections': selector.rejections,
                        'alternates': selector.alternates,
                        'retained': selector.retained, 'brakes': selector.brakes,
                        'splices': selector.splices,
                    }
                if diagnose and collision:
                    last = recent[-1]
                    last_t = last['t']
                    wp_changes = [z for a, z in zip(list(recent)[:-1], list(recent)[1:])
                                  if z['wp_pair'] != a['wp_pair']]
                    next_changes = [z for a, z in zip(list(recent)[:-1], list(recent)[1:])
                                    if z['next_pair'] != a['next_pair']]
                    recent_two = [z for z in recent if last_t-z['t'] <= 2.]
                    obs = env.dynamic_obstacles
                    d2 = np.sum((obs[:, :2] - env.boat_pos)**2, axis=1)
                    nearest = int(np.argmin(d2))
                    obstacle = obs[nearest]
                    bearing = (math.atan2(obstacle[1]-env.boat_pos[1],
                                          obstacle[0]-env.boat_pos[0])-
                               env.boat_heading+math.pi)%(2*math.pi)-math.pi
                    row['diagnosis'] = {
                        'position_px': [round(last['x'], 1), round(last['y'], 1)],
                        'speed_m_s': round(last['speed'], 3),
                        'heading_error_rad': round((last['target_heading']-last['heading']+math.pi)%(2*math.pi)-math.pi, 3),
                        'yaw_rate_rad_s': round(last['yaw'], 3),
                        'steer': round(last['steer'], 3),
                        'min_wide_px': round(last['min_wide'], 1),
                        'min_wide_last2s_px': round(min(z['min_wide'] for z in recent_two), 1),
                        'wp_pair': last['wp_pair'], 'next_pair': last['next_pair'],
                        'wp_distance_px': None if last['wp_pos'] is None else
                            round(math.dist((last['x'], last['y']), last['wp_pos']), 1),
                        'last_wp_change_age_s': None if not wp_changes else
                            round(last_t-wp_changes[-1]['t'], 2),
                        'last_next_change_age_s': None if not next_changes else
                            round(last_t-next_changes[-1]['t'], 2),
                        'wp_changes_last5s': len(wp_changes),
                        'next_changes_last5s': len(next_changes),
                        'guard_interventions_last5s': last['guard_interventions']-recent[0]['guard_interventions'],
                        'obstacle_center_px': [round(float(obstacle[0]), 1), round(float(obstacle[1]), 1)],
                        'obstacle_body_bearing_rad': round(bearing, 3),
                        'center_clearance_px': round(float(math.sqrt(d2[nearest])-obstacle[2]), 1),
                        'recent': [dict(t=z['t'], x=round(z['x'], 1), y=round(z['y'], 1),
                                        heading=round(z['heading'], 3),
                                        target_heading=round(z['target_heading'], 3),
                                        yaw=round(z['yaw'], 3),
                                        speed=round(z['speed'], 3),
                                        wp=z['wp_pair'], next=z['next_pair'],
                                        min_wide=round(z['min_wide'], 1),
                                        guard=z['guard_interventions'])
                                   for z in list(recent)[::5]],
                    }
                with output.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row)+'\n')
                rows.append(row)
                if state['seed'] == trace_seed:
                    output.with_suffix('.trace.json').write_text(json.dumps({
                        'steps': trace, 'obstacles': env.obstacles.tolist(),
                        'shadow': state['shadow_records'],
                    }, indent=2))
                if stop_on_failure and row['outcome'] != 'success':
                    raise Finished()
                if len(rows) == len(seeds):
                    raise Finished()
                state['seed'] = seeds[len(rows)]
                random.seed(state['seed'])
                np.random.seed(state['seed'])
            result = original_reset()
            state.update(active=False, last_pos=None, last_heading=None,
                         distance=0.0, turn_deg=0.0, last_sign=0,
                         reversals=0, speed_sum=0.0, steps=0,
                         shadow_records=[], shadow_cpu_s=0.0,
                         last_diff=None, command_tv=0.0,
                         straight_steps=0, straight_tv=0.0,
                         straight_yaw_sq=0.0, straight_heading_sq=0.0,
                         straight_lateral_sq=0.0, straight_last_diff=None,
                         straight_last_sign=0, straight_reversals=0,
                         portal_crossings=[], portal_switches=0,
                         last_portal_pair=None)
            recent.clear()
            return result

        def render(*args, **kwargs):
            if state['active'] and env.frame*env.dt >= timeout_s:
                if not env.collide() and np.linalg.norm(env.target-env.boat_pos) >= 70:
                    env.reset()

        env.step, env.reset, env.render = step, reset, render
        return env

    environment.EnvRenderer = _Renderer
    main.BoatEnv = factory
    main.MainSafetyGuard = guard_factory
    main.find_gap = find_gap
    main.save_episode_frame = lambda *a, **kw: None
    main.start_capture_worker = lambda: None
    main.time = FakeTime
    try:
        main.run()
    except Finished:
        pass
    finally:
        environment.EnvRenderer = original_renderer
        main.BoatEnv = original_factory
        main.MainSafetyGuard = original_guard
        main.find_gap = original_find_gap
        main.save_episode_frame = original_capture
        main.start_capture_worker = original_start_capture
        main.time = original_time
        pygame.quit()
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', required=True,
                        help='Comma-separated integers or inclusive ranges, e.g. 2000-2199')
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout-sim', type=float, default=140.0)
    parser.add_argument('--trace-seed', type=int)
    parser.add_argument('--diagnose', action='store_true')
    parser.add_argument('--collision-seeds-from')
    parser.add_argument('--shadow-horizon', type=int, default=0,
                        help='Diagnostic-only follower preview length in physics steps')
    parser.add_argument('--stop-on-failure', action='store_true')
    args = parser.parse_args()
    seeds = []
    for item in args.seeds.split(','):
        if '-' in item:
            start, end = map(int, item.split('-', 1))
            seeds.extend(range(start, end+1))
        else:
            seeds.append(int(item))
    if args.collision_seeds_from:
        seeds = [json.loads(line)['seed'] for line in Path(args.collision_seeds_from).read_text().splitlines()
                 if json.loads(line)['outcome'] == 'collision']
    result = run(seeds, args.output, args.timeout_sim, args.trace_seed,
                 args.diagnose, args.shadow_horizon, args.stop_on_failure)
    if args.stop_on_failure:
        failed = next((row for row in result if row['outcome'] != 'success'), None)
        print(f"FAIL seed {failed['seed']}: {failed['outcome']}" if failed else
              f'PASS {len(result)}/{len(seeds)}')
        sys.exit(1 if failed else 0)
    print(json.dumps({kind: sum(row['outcome']==kind for row in result)
                      for kind in ('success', 'collision', 'timeout')}))
