"""Render-only arc-length interpolation on the existing displayed Bezier."""
import numpy as np


class PursuitDisplayMarker:
    def __init__(self):
        self.reset()

    def reset(self):
        self.points = self.point = None
        self.last_time = self.published_time = None
        self.seen_generation = None
        self.period = None
        self.stop_point = None
        self.at_stop = False
        self.path_continuous = False

    def _project(self, point):
        fraction = np.clip(np.sum((point-self.points[:-1])*self.segments, axis=1)/
                           np.maximum(self.lengths**2, 1e-12), 0., 1.)
        projected = self.points[:-1]+fraction[:, None]*self.segments
        distance2 = np.sum((projected-point)**2, axis=1)
        index = int(np.argmin(distance2))
        return (self.arc[index]+fraction[index]*self.lengths[index],
                float(np.sqrt(distance2[index])))

    def _evaluate(self, progress):
        progress = min(max(0., float(progress)), self.arc[-1])
        index = min(max(0, int(np.searchsorted(self.arc, progress, side='right'))-1),
                    len(self.points)-2)
        fraction = (progress-self.arc[index])/max(self.lengths[index], 1e-12)
        return self.points[index]+fraction*self.segments[index]

    def hold_at_stop(self):
        """Latch only the displayed marker to the current path intersection."""
        if self.stop_point is not None:
            self.progress = self.stop_arc
            self.point = self.stop_point.copy()
            self.at_stop = True

    def set_path(self, path, target, generation, speed, cadence_sim_s, scale, stop=None):
        """Cache/project on publication, never regenerate or modify the path."""
        if path is None or target is None or len(path) < 2:
            self.reset()
            return
        prior = self.point
        self.points = np.asarray(path)
        self.segments = np.diff(self.points, axis=0)
        self.lengths = np.linalg.norm(self.segments, axis=1)
        self.arc = np.r_[0., np.cumsum(self.lengths)]
        self.target_arc, _ = self._project(np.asarray(target))
        self.stop_point = None if stop is None else np.asarray(stop).copy()
        self.stop_arc = (self.arc[-1] if stop is None else self._project(self.stop_point)[0])
        self.target_arc = min(self.target_arc,self.stop_arc)
        self.progress = self.target_arc
        self.path_continuous = prior is None
        if prior is not None:
            progress, distance = self._project(prior)
            # Like CODEX's marker anchor: retain a nearby projection; a truly
            # different route adopts its current target immediately.
            nearby = 2.*max(.1*scale, speed*cadence_sim_s)
            if distance <= nearby and abs(progress-self.target_arc) <= 2.*nearby:
                self.progress = progress
                self.path_continuous = True
        self.progress = min(self.progress,self.stop_arc)
        self.point = self._evaluate(self.progress)
        self.at_stop = self.stop_point is not None and self.progress>=self.stop_arc
        if self.at_stop:
            self.point = self.stop_point.copy()
        self.generation = generation
        self.cadence_sim_s = cadence_sim_s

    def update(self, now, speed, physics_dt, paused=False):
        """Advance every render frame, with no x/y filtering or corner shortcut."""
        if self.point is None:
            return None
        elapsed = 0. if self.last_time is None else max(0., now-self.last_time)
        self.last_time = now
        if paused:
            self.published_time = now
            return self.point.copy()
        if self.generation != self.seen_generation:
            if self.seen_generation is not None and self.published_time is not None:
                advanced = (self.generation-self.seen_generation)*physics_dt
                if advanced > 0. and now > self.published_time:
                    self.period = self.cadence_sim_s*(now-self.published_time)/advanced
            self.seen_generation = self.generation
            self.published_time = now
        period = max(1e-6, self.period or self.cadence_sim_s)
        # Infer actual display progression from publication time. This follows
        # the achieved playback rate without changing/duplicating its mapping.
        speed_wall = max(0., speed)*self.cadence_sim_s/period
        age = min(period, max(0., now-self.published_time))
        desired = min(self.stop_arc, self.target_arc+speed_wall*age)
        error = max(0., desired-self.progress)
        rate = max(speed_wall, error/period)
        self.progress += min(error, rate*elapsed)
        self.point = self._evaluate(self.progress)
        self.at_stop = self.stop_point is not None and self.progress>=self.stop_arc
        if self.at_stop:
            self.point = self.stop_point.copy()
        return self.point.copy()
