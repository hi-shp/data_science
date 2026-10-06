"""Bounded precomputation of the unchanged fixed-step autonomous pipeline.

The model has no SDL, rendering, wake RNG or wall-clock pacing. The GUI applies
every packet in order with BoatEnv.step and verifies its authoritative state.
"""
import atexit
import copy
import multiprocessing as mp
import queue
from collections import deque
from types import SimpleNamespace

import numpy as np

from environment import BoatEnv
from trajectory_runtime import advance_trajectory
from perception import update_grid


FIELDS = ('frame', 'boat_pos', 'boat_vel', 'boat_heading', 'boat_ang_vel',
          'thrust_left', 'thrust_right', 'obstacles', 'dynamic_obstacles',
          'grid', 'rel_angles', 'lidar_range', 'map_w', 'sim_h', 'dt',
          'dynamics', 'control', 'navigation_mode', 'command_speed',
          'command_yaw_rate', 'target', 'navigation_map', 'trajectory_navigator',
          'heading_target', 'min_wide_dist', 'prediction_frame', '_line_resume_plan')
DISPLAY_FIELDS = ('raw_route', 'control_path', 'predicted_trajectory',
                  'controller_target', 'pursuit_target', 'predicted_clearance',
                  'prediction_stride_steps', 'perceived_obstacles',
                  'navigation_map', 'trajectory_navigator', 'selected_gap',
                  'current_wp', 'next_wp', 'all_gaps', 'total_gaps_count')


def snapshot(env):
    return copy.deepcopy({name: getattr(env, name) for name in FIELDS
                          if hasattr(env, name)})


class _Model(SimpleNamespace):
    physics_state = BoatEnv.physics_state
    get_pwm = BoatEnv.get_pwm
    pwm_to_thrust = BoatEnv.pwm_to_thrust
    update_dynamic_obstacles = BoatEnv.update_dynamic_obstacles
    _physics_step = BoatEnv.step

    def __init__(self, values):
        super().__init__(**values)
        self.headless = True
        self.manual_mode = self.linetrace_mode = False
        self.pwm = None

    def step(self, left, right, **kwargs):
        self.pwm = left, right
        self._physics_step(left, right, **kwargs)

    def update_camera(self):
        pass


class TrajectoryWorker:
    def __init__(self, env, capacity=12):
        self.capacity = capacity
        context = mp.get_context('spawn')
        self.requests = context.Queue()
        self.results = context.Queue(maxsize=capacity)
        self.stop = context.Event()
        self.ready = deque()
        self.epoch = 0
        self.process = context.Process(target=self._run,
                                       args=(snapshot(env), self.requests,
                                             self.results, self.stop),
                                       name='codex-fixed-step-control', daemon=True)
        self.process.start()
        atexit.register(self.close)

    @staticmethod
    def _run(values, requests, results, stop):
        epoch, model = 0, _Model(values)
        try:
            while not stop.is_set():
                try:
                    epoch, values = requests.get_nowait()
                    model = _Model(values)
                except queue.Empty:
                    pass
                hits = advance_trajectory(model)
                packet = dict(epoch=epoch, frame=model.frame, pwm=model.pwm,
                              state=model.physics_state(), hits=hits,
                              lidar_dists=model.lidar_dists,
                              prev_steer=model.prev_steer,
                              command_speed=model.command_speed,
                              command_yaw_rate=model.command_yaw_rate,
                              heading_target=model.heading_target,
                              min_wide_dist=model.min_wide_dist,
                              prediction_frame=model.prediction_frame)
                if model.frame == model.prediction_frame:
                    packet['prediction'] = copy.deepcopy({name: getattr(model, name)
                                                         for name in DISPLAY_FIELDS})
                while not stop.is_set():
                    try:
                        results.put(packet, timeout=.002)
                        break
                    except queue.Full:
                        if not requests.empty():
                            break  # explicit reset/mode change, never a lost physics tick
        except BaseException as error:
            results.put({'error': repr(error)}, timeout=1.)
            # Queue.put uses a feeder thread. Flush diagnostics before exit so
            # the GUI receives the original failure instead of only "stopped".
            results.close()
            results.join_thread()
        finally:
            results.cancel_join_thread()

    def reset(self, env):
        self.epoch += 1
        self.ready.clear()
        self.requests.put((self.epoch, snapshot(env)))

    def available(self):
        while len(self.ready) < self.capacity:
            try:
                packet = self.results.get_nowait()
            except queue.Empty:
                break
            if 'error' in packet:
                raise RuntimeError('CODEX control worker failed: '+packet['error'])
            if packet['epoch'] == self.epoch:
                self.ready.append(packet)
        if not self.process.is_alive() and not self.stop.is_set():
            raise RuntimeError('CODEX control worker stopped')
        return len(self.ready)

    def prime(self):
        while not self.available():
            self.stop.wait(.001)

    def advance(self, env, step_idx=0, sub_steps=1):
        packet = self.ready.popleft()
        if packet['frame'] != env.frame+1:
            raise RuntimeError('CODEX command belongs to a different physics tick')
        env.frame += 1
        env.update_dynamic_obstacles()
        if 'prediction' in packet:
            env.__dict__.update(packet['prediction'])
        for name in ('lidar_dists', 'prev_steer', 'command_speed',
                     'command_yaw_rate', 'heading_target', 'min_wide_dist',
                     'prediction_frame'):
            setattr(env, name, packet[name])
        # Replay the original two cheap grid operations on the same scan. This
        # avoids serializing the 423 KB display grid at every physics tick.
        update_grid(env.grid, *packet['hits'])
        env.grid *= .945
        env.step(*packet['pwm'], sub_step_idx=step_idx, total_sub_steps=sub_steps)
        if not np.array_equal(env.physics_state(), packet['state']):
            raise RuntimeError('CODEX worker diverged from authoritative physics')
        env.update_camera()
        return packet['hits']

    def close(self):
        if not self.stop.is_set():
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
