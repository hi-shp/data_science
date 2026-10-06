"""Bounded CODEX command computation for the MAIN-style 2D display.

No SDL, rendering, wake simulation, or physics-time pacing runs in this process.
The GUI applies every completed command to authoritative BoatEnv.step once.
"""
import atexit
import math
import multiprocessing as mp
import queue
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
from heavy.motion_core.trajectory_runtime import advance_trajectory
from vessel_dynamics import integrate


FIELDS = ('frame', 'boat_pos', 'boat_vel', 'boat_heading', 'boat_ang_vel',
          'thrust_left', 'thrust_right', 'obstacles', 'dynamic_obstacles',
          'grid', 'rel_angles', 'lidar_range', 'map_w', 'sim_h', 'dt',
          'dynamics', 'control', 'navigation_mode', 'command_speed', 'command_yaw_rate')


def snapshot(env):
    return {key: (getattr(env,key).copy() if isinstance(getattr(env,key),np.ndarray)
                  else getattr(env,key)) for key in FIELDS} | {'target':env.target.copy(), '_line_resume_plan':getattr(env, '_line_resume_plan', False)}


class _Model(SimpleNamespace):
    # Reuse the exact CODEX-adapted environment allocation/state functions.
    from environment import BoatEnv as _BoatEnv
    physics_state = _BoatEnv.physics_state
    get_pwm = _BoatEnv.get_pwm
    pwm_to_thrust = _BoatEnv.pwm_to_thrust
    update_dynamic_obstacles = _BoatEnv.update_dynamic_obstacles

    def __init__(self, values):
        super().__init__(**values)
        self.motion_core_v2 = True
        self.manual_mode = self.linetrace_mode = False
        self.reflected_wakes = []
        self.pwm = None

    def step(self, left, right, **kwargs):
        # Identical numerical portion of BoatEnv.step; no display-side work.
        z = integrate(self.physics_state(), self.pwm_to_thrust(left),
                      self.pwm_to_thrust(right), self.dt, self.dynamics)
        scale = self.dynamics.pixels_per_m
        self.boat_pos = z[:2]*scale
        self.boat_heading = float(z[2])
        c,s = math.cos(z[2]),math.sin(z[2])
        self.boat_vel = np.array([z[3]*c-z[4]*s,z[3]*s+z[4]*c])*scale
        self.boat_ang_vel = float(z[5])
        self.thrust_left,self.thrust_right = float(z[6]),float(z[7])
        self.pwm = float(left),float(right)

    def update_camera(self):
        pass


def _run(values, requests, results, stop):
    epoch, model = 0, _Model(values)
    from heavy.motion_v2 import warmup
    warmup(model)
    try:
        while not stop.is_set():
            try:
                epoch, values = requests.get_nowait()
                model = _Model(values)
            except queue.Empty:
                pass
            hx,hy = advance_trajectory(model)
            model.reflected_wakes.clear()
            packet = dict(epoch=epoch,frame=model.frame,pwm=model.pwm,state=model.physics_state(),
                hits=(hx,hy),lidar_dists=model.lidar_dists,prev_steer=model.prev_steer,
                command_speed=model.command_speed,command_yaw_rate=model.command_yaw_rate,
                heading_target=model.heading_target,min_wide_dist=model.min_wide_dist,
                prediction_frame=model.prediction_frame)
            if model.frame == model.prediction_frame:
                packet['prediction'] = dict(raw_route=model.raw_route,control_path=model.control_path,
                    predicted_trajectory=model.predicted_trajectory,controller_target=model.controller_target,
                    motion_prediction_states=model.motion_prediction_states,
                    predicted_clearance=model.predicted_clearance,
                    prediction_stride_steps=model.prediction_stride_steps,
                    perceived_obstacles=model.perceived_obstacles,
                    navigation_map=SimpleNamespace(obstacles=model.navigation_map.obstacles.copy()),
                    trajectory_navigator=SimpleNamespace(path_frame=model.trajectory_navigator.path_frame),
                    motion_plan_ms=model.trajectory_navigator.timings['plan'][-1]*1000.)
            while not stop.is_set():
                try:
                    results.put(packet,timeout=.002)
                    break
                except queue.Full:
                    if not requests.empty():
                        break  # abandon stale work only at an explicit episode reset
    except BaseException as error:
        results.put({'error':repr(error)},timeout=1.)
    finally:
        results.cancel_join_thread()


class MotionCommandWorker:
    def __init__(self, env, capacity=12):
        context = mp.get_context('spawn')
        self.requests = context.Queue()
        self.results = context.Queue(maxsize=capacity)
        self.stop = context.Event()
        self.epoch = 0
        self.ready = deque()
        self.capacity = capacity
        self.process = context.Process(target=_run,args=(snapshot(env),self.requests,self.results,self.stop),
                                       name='main-heavy-2d-control',daemon=True)
        self.process.start()
        atexit.register(self.close)

    def reset(self, env):
        self.epoch += 1
        self.ready.clear()
        self.requests.put((self.epoch,snapshot(env)))

    def available(self):
        while len(self.ready)<self.capacity:
            try:
                packet = self.results.get_nowait()
            except queue.Empty:
                break
            if 'error' in packet:
                raise RuntimeError(packet['error'])
            if packet['epoch'] == self.epoch:
                self.ready.append(packet)
        if not self.process.is_alive() and not self.stop.is_set():
            raise RuntimeError('MAIN_HEAVY control process stopped unexpectedly')
        return len(self.ready)

    def take(self):
        return self.ready.popleft()

    def prime(self):
        while not self.available():
            time.sleep(.001)

    def close(self):
        if self.stop.is_set():
            return
        self.stop.set()
        self.process.join(timeout=.25)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=.25)
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=.25)
        self.requests.cancel_join_thread()
        self.results.cancel_join_thread()
        self.requests.close()
        self.results.close()
        atexit.unregister(self.close)
