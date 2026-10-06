"""Small, resumable Phase 5 gate; stop on the first collision or timeout."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from pathlib import Path

from evaluate_main_heavy import run


GATES = (
    ('difficult', [2000, 2069, 2081, 2094, 2189]),
    ('regression', [2020, 2163, 2177, 2035, 2037]),
    ('normal', [2001, 2002, 2006, 2010, 2019,
                2025, 2033, 2024, 2026, 2041]),
    ('straight', [2003, 2004, 2005, 2008, 2009]),
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', required=True)
    parser.add_argument('--output-dir', default='data/main_heavy')
    parser.add_argument('--timeout-sim', type=float, default=140.0)
    args = parser.parse_args()
    folder = Path(args.output_dir)
    completed = 0
    for name, seeds in GATES:
        rows = run(seeds, folder / f'{args.tag}_{name}.jsonl',
                   args.timeout_sim, stop_on_failure=True)
        failed = next((row for row in rows if row['outcome'] != 'success'), None)
        if failed is not None:
            print(f"FAIL seed {failed['seed']}: {failed['outcome']} ({name})")
            return 1
        completed += len(rows)
    print(f'PASS {completed}/{sum(len(seeds) for _, seeds in GATES)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
