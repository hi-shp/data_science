"""Unchanged CODEX control configuration; legacy controller is not imported."""
import math
from dataclasses import dataclass

@dataclass(frozen=True)
class ControllerParameters:
    planning_period_steps: int = 3  # 0.12 s, independent of display frame rate
    horizon_s: float = 5.0
    prediction_step_s: float = 0.2
    lookahead_m: float = 3.0
    safety_margin_m: float = 0.20
    target_wall_margin_m: float = 1.8
    yaw_samples: int = 21
    heading_gain: float = 0.65
    cross_track_weight: float = 0.18
    turning_weight: float = 0.22
    command_change_weight: float = 3.0
    terminal_heading_weight: float = 0.40
    goal_distance_weight: float = 0.20

    def __post_init__(self):
        for name in self.__dataclass_fields__:
            value = getattr(self,name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.prediction_step_s > self.horizon_s:
            raise ValueError('prediction_step_s must not exceed horizon_s')
        if self.planning_period_steps != int(self.planning_period_steps) or self.yaw_samples != int(self.yaw_samples):
            raise ValueError('planning_period_steps and yaw_samples must be integers')

