"""Named, opt-in trajectory-controller comparisons for the GUI."""
from dataclasses import replace

from experiments.sampling_navigation import SamplingConfig


MODE_FLAGS = {
    'eta_base': (),
    'eta_value': ('terminal_value',),
    'eta_continuity': ('continuity',),
    'eta_continuity_goal': ('continuity', 'strategy_guidance',
                            'goal_entry_heading', 'center_entry'),
    'eta_continuity_passage': ('continuity', 'strategy_guidance',
                               'goal_entry_heading', 'center_entry', 'passage_guidance'),
    'eta_continuity_passage_exact': ('continuity', 'strategy_guidance',
                                     'goal_entry_heading', 'center_entry',
                                     'passage_guidance', 'exact_passage_safety'),
    'eta_continuity_forward': ('continuity', 'strategy_guidance',
                               'goal_entry_heading', 'center_entry',
                               'passage_guidance', 'exact_passage_safety',
                               'forward_policy'),
    'eta_multimodal': ('multimodal',),
    'eta_viability': ('viability',),
    'eta_recovery': ('recovery_policy',),
    'eta_full': ('terminal_value', 'continuity', 'multimodal',
                 'viability', 'recovery_policy'),
}
NAV_MODES = (*MODE_FLAGS, 'legacy_a', 'trajectory_reference')


def config_for_mode(mode):
    if mode not in MODE_FLAGS:
        raise ValueError(f'Unknown ETA mode: {mode}')
    base = SamplingConfig(smooth_weight=3., yaw_command_step=.1,
                          symmetric_yaw_bias=0., objective='eta')
    changes = {flag: True for flag in MODE_FLAGS[mode]}
    if mode in ('eta_continuity_goal', 'eta_continuity_passage',
                'eta_continuity_passage_exact', 'eta_continuity_forward'):
        changes['goal_radius_m'] = .55
    return replace(base, **changes)
