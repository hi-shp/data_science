"""Observe the actual X11 production worker/scheduler; never substitute physics.

Example: python3 experiments/regression/measure_motion_gui.py --branch main
--root . --rate 4 --seed 2069 --seconds 12 --output scratch/gui.json
Use an exported CODEX tree as --root for the reference. Only capture-file writes
are suppressed. GUI, wake and planning workload stay enabled.
"""
import argparse
from collections import deque
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np


def run(root, branch, rate, seed, seconds, output, episode=False):
    if not os.environ.get('DISPLAY') or os.environ.get('SDL_VIDEODRIVER')=='dummy':
        raise RuntimeError('Real X11 is required')
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT']='1'
    sys.path.insert(0,str(root)); os.chdir(root)
    import main
    import pygame
    # Workers bind the actual class before the instrumentation factory replaces it.
    if branch=='main':
        import heavy.motion_worker
    else:
        import trajectory_worker
    factory,events,budget=main.BoatEnv,pygame.event.get,main.playback_budget
    frames,backlogs,rows,plans,predictions=[],[],[],[],[]
    timing=dict(start=None,end=None,steps=0,budget=0.,batch=0,done=False,captured=False,goal_captured=False)
    def make_env():
        random.seed(seed);np.random.seed(seed)
        env=factory();env.sim_speed=rate;env.fullscreen_3d=False
        step,render,reset=env.step,env.render,env.reset
        def observed_step(l,r,**kwargs):
            result=step(l,r,**kwargs)
            if timing['start'] is not None:
                timing['steps']+=1;timing['batch']+=1
            rows.append([env.frame,env.frame*env.dt,*env.physics_state(),
                env.command_speed,env.command_yaw_rate,l,r,
                float(np.linalg.norm(env.boat_pos-env.target)),env.prediction_frame,
                int(env.collide())])
            predictions.append(hashlib.sha256(env.control_path.tobytes()).hexdigest())
            if env.frame==env.prediction_frame:
                if branch=='main': plans.append(env.motion_plan_ms)
                else: plans.append(env.trajectory_navigator.timings['plan'][-1]*1000)
            return result
        def observed_render(*args):
            result=render(*args);now=time.perf_counter()
            if (episode and branch=='main' and not timing['goal_captured'] and
                    getattr(env,'goal_display_connected',False) and
                    np.linalg.norm(env.boat_pos-env.target)<200.):
                pygame.image.save(env.screen,str(output.with_name(output.stem+'_goal.png')))
                timing['goal_captured']=True
            if timing['start'] is None:timing['start']=now
            else:
                frames.append(now);backlogs.append(max(0.,timing['budget']-timing['batch']*env.dt)/env.dt)
            timing['batch']=0;timing['end']=now
            if not timing['captured'] and ((episode and timing['done']) or
                (not episode and now-timing['start']>=seconds)):
                pygame.image.save(env.screen,str(output.with_suffix('.png')))
                timing['captured']=True
            return result
        def observed_reset():
            timing['done']=True
            return reset()
        env.step,env.render,env.reset=observed_step,observed_render,observed_reset
        return env
    def observed_budget(*args):
        timing['budget']=budget(*args);return timing['budget']
    def observed_events(*args):
        result=events(*args)
        if timing['start'] is not None and (timing['done'] if episode else
            time.perf_counter()-timing['start']>=seconds):result.append(pygame.event.Event(pygame.QUIT))
        return result
    main.BoatEnv=make_env;main.playback_budget=observed_budget
    main.save_episode_frame=lambda *a,**k:None;main.start_capture_worker=lambda:None
    pygame.event.get=observed_events
    main.run()
    wall=timing['end']-timing['start']; intervals=np.diff(frames)*1000
    window=deque();rolling=[]
    for t in frames:
        window.append(t)
        while window[0]<t-1:window.popleft()
        if t-frames[0]>=1:rolling.append(len(window))
    result=dict(branch=branch,seed=seed,rate=rate,wall_s=wall,fps=len(frames)/wall,
        rolling_min=min(rolling),p95_ms=float(np.percentile(intervals,95)),
        p99_ms=float(np.percentile(intervals,99)),stalls_100ms=int(np.sum(intervals>=100)),
        physics_s=timing['steps']/wall,requested_physics_s=2.4*rate/.04,
        backlog_max_steps=max(backlogs),backlog_final_steps=backlogs[-1],
        planning_median_ms=float(np.median(plans)),physics_steps=len(rows))
    output.write_text(json.dumps(result,indent=2));np.save(output.with_suffix('.npy'),rows)
    output.with_suffix('.predictions.json').write_text(json.dumps(predictions))
    print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path.cwd())
    p.add_argument('--branch',choices=['main','codex'],default='main')
    p.add_argument('--rate',type=int,default=4);p.add_argument('--seed',type=int,default=2069)
    p.add_argument('--seconds',type=float,default=12);p.add_argument('--episode',action='store_true')
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output=a.output.resolve();a.output.parent.mkdir(parents=True,exist_ok=True)
    run(a.root.resolve(),a.branch,a.rate,a.seed,a.seconds,a.output,a.episode)
