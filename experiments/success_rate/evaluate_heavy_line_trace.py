"""Line Tracing now evaluates MAIN compatibility, not heavy-specific retuning.

Historical nearest-hit/low-speed traces remain under data/main_heavy/line_trace.
Use evaluate_main_line_compat.py for current per-step input/command/state parity.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.success_rate.evaluate_main_line_compat import evaluate
import argparse
if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--seeds',default='2000-2199')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();seeds=[]
    for chunk in args.seeds.split(','):
        if '-' in chunk:
            first,last=map(int,chunk.split('-'));seeds.extend(range(first,last+1))
        else:seeds.append(int(chunk))
    evaluate(seeds,args.output)
