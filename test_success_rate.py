"""Evaluate the same fixed-step controller used by main.py.

python3 test_success_rate.py 100 --headless --seed 2000
"""
import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes', nargs='?', type=int, default=10000)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--seed', type=int, default=2000)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--width', type=int, default=1800)
    parser.add_argument('--timeout', type=float, default=140.)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or Path('data') / ('evaluation_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    command = [sys.executable, str(Path(__file__).with_name('benchmark_navigation.py')),
               '--episodes', str(args.episodes), '--seed', str(args.seed),
               '--workers', str(args.workers), '--width', str(args.width),
               '--timeout', str(args.timeout), '--output', str(output)]
    if not args.headless:
        command.append('--render')
    return subprocess.call(command)


if __name__ == '__main__':
    raise SystemExit(main())
