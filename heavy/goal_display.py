"""MAIN's independent goal guide uses the shared pure display policy."""
from heavy.motion_core.goal_guidance import GOAL_ZONE_M
from heavy.gap_annotation import BEZIER_CONTROL_DEVIATION_PX
from goal_display_policy import corridor_clear, safe_goal_path


def goal_guidance_path(*args, **kwargs):
    return safe_goal_path(*args, **kwargs, goal_zone_m=GOAL_ZONE_M,
                          reserve=BEZIER_CONTROL_DEVIATION_PX)
