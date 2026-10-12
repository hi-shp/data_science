"""Check every recorded production render, including brief unsafe transitions.

Run against measure_motion_gui.py --trace-display outputs. Physics traces
remain separate and can be compared with compare_motion_branches.py.
Suppressed unsafe guidance is permitted; a connected, off-goal endpoint is not.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def analyze(path):
    rows=json.loads(path.read_text())
    connected=[r for r in rows if r['connected']]
    errors=[float(np.linalg.norm(np.asarray(r['rendered_endpoint'])-r['goal']))
            for r in connected]
    if any(e>1e-6 for e in errors):raise AssertionError(f'{path}: inexact connected goal')
    bad=[];suppressed=0
    for before,after in zip(rows,rows[1:]):
        # Safety-driven suppression is deliberately not a visual continuity
        # failure, and must not be replaced with the previous safe connector.
        if before['connected'] and not after['connected']:
            if after.get('reason','').startswith('GUIDANCE_SUPPRESSED_'):
                if len(after['rendered_path'])>1:
                    raise AssertionError(f'{path}: stale unsafe guide is still drawn')
                suppressed+=1
            else:bad.append(after['render_frame'])
    return dict(frames=len(rows),connected_frames=len(connected),
        max_goal_error_px=max(errors or [0.]),unsafe_guidance_suppressed=suppressed,
        unwanted_goal_to_finite_switches=bad)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('traces',nargs='+',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result={str(path):analyze(path) for path in args.traces}
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(frames=sum(v['frames'] for v in result.values()),
        connected_frames=sum(v['connected_frames'] for v in result.values()),
        unwanted_switches=sum(len(v['unwanted_goal_to_finite_switches']) for v in result.values()),
        max_goal_error_px=max(v['max_goal_error_px'] for v in result.values())),indent=2))
    if any(v['unwanted_goal_to_finite_switches'] for v in result.values()):
        raise SystemExit('Unexpected goal-to-finite display transition')
