"""Compare pinned MAIN production methods to branch Line Tracing, per step.

No Git checkout/worktree is needed for the MAIN reference: its source is loaded
from the pinned commit in memory. This is simulation-state regression testing,
not a wall-clock leaderboard measurement. Renderers are replaced for the seed
batch; physics, sensors, obstacle movement, commands and contact are unchanged.
"""
import os
os.environ.setdefault('SDL_VIDEODRIVER', 'dummy')
os.environ.setdefault('SDL_AUDIODRIVER', 'dummy')
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import argparse
import ast
import hashlib
import json
import math
import random
import subprocess
import sys
from pathlib import Path
from types import ModuleType
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pygame
import environment
import main_line_compat as compat

class NoRenderer:
    def __init__(self, env): self.engine_3d = None
    def render(self, *args): pass

def source(file):
    return subprocess.check_output(['git', 'show', f'{compat.MAIN_REFERENCE_COMMIT}:{file}'], text=True)

def reference():
    nav = ModuleType('_main_line_reference_navigation')
    exec(compile(source('navigation.py'), 'MAIN/navigation.py', 'exec'), nav.__dict__)
    envmod = ModuleType('_main_line_reference_environment')
    # MAIN imports reactive_avoidance, which CODEX's navigation module omits.
    # Bind the reference's own navigation dependencies during source loading.
    saved_navigation = sys.modules.get('navigation')
    sys.modules['navigation'] = nav
    try:
        exec(compile(source('environment.py'), 'MAIN/environment.py', 'exec'), envmod.__dict__)
    finally:
        if saved_navigation is None:
            del sys.modules['navigation']
        else:
            sys.modules['navigation'] = saved_navigation
    envmod.EnvRenderer = NoRenderer
    sensor = ModuleType('_main_line_reference_perception')
    exec(compile(source('perception.py'), 'MAIN/perception.py', 'exec'), sensor.__dict__)
    return envmod.BoatEnv, nav.line_trace_steering, sensor.lidar_hits_np

def verify_pinned_sources():
    for file, names in (('environment.py', ('get_pwm','pwm_to_thrust','step','collide')),
                        ('navigation.py', ('line_trace_steering',)),
                        ('perception.py', ('lidar_hits_np',))):
        original={n.name:ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(source(file))) if isinstance(n, ast.FunctionDef) and n.name in names}
        actual={n.name:ast.dump(n, include_attributes=False) for n in ast.walk(ast.parse(Path(compat.__file__).read_text())) if isinstance(n, ast.FunctionDef) and n.name in names}
        assert original == actual, (file, 'MAIN implementation drift')

def episode(seed, cls, controller, sensor, timeout=140., keep=False):
    random.seed(seed); np.random.seed(seed)
    env=cls()
    if hasattr(env, 'set_line_tracing'):
        random.seed(seed); np.random.seed(seed)
        env.linetrace_mode=True
        env.reset()
    else:
        env.linetrace_mode=True
    digest=hashlib.sha256(); states=[]; inputs=[]
    map_hash=hashlib.sha256(env.obstacles.tobytes()).hexdigest()
    for tick in range(int(timeout/env.dt)):
        env.frame+=1;env.update_dynamic_obstacles()
        dists,hx,hy=sensor(env.boat_pos,env.boat_heading,env.rel_angles,
                           env.dynamic_obstacles,env.lidar_range,
                           map_bounds=(0,0,env.map_w,env.sim_h))
        env.lidar_dists=dists
        steer,heading,distance,hit=controller(env.boat_pos,env.boat_heading,env.target,
                                              dists,env.rel_angles,env.boat_ang_vel,env.prev_steer)
        env.prev_steer=steer;env.heading_target=heading
        env.min_wide_dist=distance;env.closest_avoid_hit=hit
        left,right=env.get_pwm(steer)
        env.step(left,right)
        collision=bool(env.collide())
        reached=math.hypot(*(env.target-env.boat_pos)) < 70.
        values=np.asarray([*env.boat_pos,env.boat_heading,*env.boat_vel,env.boat_ang_vel,
                           env.current_fwd,left,right,steer,heading,distance,collision,reached],dtype=np.float64)
        digest.update(values.tobytes());digest.update(dists.tobytes())
        digest.update(np.asarray(hit if hit is not None else [np.nan,np.nan],dtype=np.float64).tobytes())
        if keep:states.append(values);inputs.append(dists.copy())
        if collision or reached:break
    result=dict(seed=seed,steps=tick+1,outcome='collision' if collision else 'success' if reached else 'timeout',
                simulation_s=(tick+1)*env.dt,state_commands_sensor_hash=digest.hexdigest(),map_hash=map_hash)
    if keep:result.update(states=np.array(states),inputs=np.array(inputs))
    pygame.quit()
    return result

def evaluate(seeds, output, reference_results=None):
    verify_pinned_sources()
    refcls,refcontrol,refsensor=reference()
    environment.EnvRenderer=NoRenderer
    from navigation import line_trace_steering
    branch=subprocess.check_output(['git','branch','--show-current'],text=True).strip()
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    cached = None
    if reference_results:
        cached_rows = [json.loads(line) for line in Path(reference_results).read_text().splitlines()]
        metadata = json.loads(Path(str(reference_results)+'.summary.json').read_text())
        assert metadata['reference_commit'] == compat.MAIN_REFERENCE_COMMIT
        assert metadata['exact_parity'] and [r['reference']['seed'] for r in cached_rows] == seeds
        cached = {r['reference']['seed']: r['reference'] for r in cached_rows}
    for seed in seeds:
        keep=seed in (2000,2069,2081)
        before=(cached[seed].copy() if cached else
                episode(seed,refcls,refcontrol,refsensor,keep=keep))
        after=episode(seed,environment.BoatEnv,line_trace_steering,compat.lidar_hits_np,keep=keep)
        for field in ('map_hash','state_commands_sensor_hash','steps','outcome'):
            assert before[field] == after[field], (branch,seed,field,before[field],after[field])
        if keep:
            if cached:
                recorded=np.load(Path(reference_results).with_name(f"{metadata['branch']}_{seed}_trajectory.npz"))
                np.testing.assert_array_equal(recorded['states'],after['states'])
                np.testing.assert_array_equal(recorded['inputs'],after['inputs'])
            else:
                np.testing.assert_array_equal(before.pop('states'),after['states'])
                np.testing.assert_array_equal(before.pop('inputs'),after['inputs'])
            np.savez_compressed(path.with_name(f'{branch}_{seed}_trajectory.npz'),states=after.pop('states'),inputs=after.pop('inputs'))
        rows.append(dict(reference=before,branch=after))
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary=dict(branch=branch,reference_commit=compat.MAIN_REFERENCE_COMMIT,episodes=len(rows),exact_parity=True,
                 outcomes={k:sum(r['branch']['outcome']==k for r in rows) for k in ('success','collision','timeout')},
                 verified_physics_steps=sum(r['branch']['steps'] for r in rows))
    Path(str(path)+'.summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--seeds',default='2000-2199');parser.add_argument('--output',required=True)
    parser.add_argument('--reference-results')
    args=parser.parse_args();seeds=[]
    for chunk in args.seeds.split(','):
        if '-' in chunk:
            a,b=map(int,chunk.split('-'));seeds.extend(range(a,b+1))
        else:seeds.append(int(chunk))
    evaluate(seeds,args.output,args.reference_results)
