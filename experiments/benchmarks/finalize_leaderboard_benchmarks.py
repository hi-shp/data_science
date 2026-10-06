"""Freeze two paired, completed visible-GUI runs into four leaderboard rows."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import json
from pathlib import Path


def read_run(directory):
    metadata = json.loads((directory / 'metadata.json').read_text())
    rows = [json.loads(line) for line in
            (directory / 'episodes.jsonl').read_text().splitlines()]
    if (metadata['episodes'] != 1000 or len(rows) != 1000 or
            len({row['seed'] for row in rows}) != 1000 or
            metadata['measurement'] != 'visible_X11_3D_main_run_perf_counter' or
            metadata['entrypoint'] != 'main.run' or
            metadata['display_speed'] != '1x' or
            not (directory / 'summary.json').exists()):
        raise ValueError(f'Incomplete or non-GUI 1000-run benchmark: {directory}')
    return metadata, sorted(rows, key=lambda row: row['seed'])


def make_records(label, rows):
    completed = [row for row in rows if row['outcome'] == 'success']
    if not completed:
        raise ValueError(f'{label}: no successful arrival')
    best = min(completed, key=lambda row: row['time_s'])
    collision_average = sum(row['outcome'] == 'collision' for row in rows) / len(rows)
    return [
        {'type': 'benchmark', 'benchmark_id': f'{label}_avg',
         'player': f'{label.upper()} AVG',
         'collisions': round(collision_average, 4),
         'time': round(sum(row['time_s'] for row in completed) / len(completed), 4),
         'cumulative_turn_deg': round(
             sum(row['cumulative_turn_deg'] for row in completed) / len(completed), 1),
         'date': '1000 GUI runs', 'successes': len(completed)},
        {'type': 'benchmark', 'benchmark_id': f'{label}_best',
         'player': f'{label.upper()} BEST', 'collisions': 0,
         'time': best['time_s'], 'cumulative_turn_deg': best['cumulative_turn_deg'],
         'date': '1000 GUI runs', 'best_seed': best['seed']},
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, required=True)
    parser.add_argument('--codex', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main_meta, main_rows = read_run(args.main)
    codex_meta, codex_rows = read_run(args.codex)
    if (main_meta['seed_start'] != codex_meta['seed_start'] or
            main_meta['window_width_px'] != codex_meta['window_width_px'] or
            main_meta['window_height_px'] != codex_meta['window_height_px'] or
            [row['seed'] for row in main_rows] != [row['seed'] for row in codex_rows] or
            [row['map_hash'] for row in main_rows] !=
            [row['map_hash'] for row in codex_rows]):
        raise ValueError('MAIN and CODEX were not measured on identical maps')
    document = {
        'schema_version': 1,
        'measurement': 'Visible X11/3D main.run perf_counter seconds at displayed 1x',
        'time_turn_aggregation': 'successful arrivals only',
        'collision_aggregation': 'all 1000 episodes, including failures',
        'seed_start': main_meta['seed_start'],
        'episodes_per_version': 1000,
        'source_commits': {'main': main_meta['source_commit'],
                           'codex': codex_meta['source_commit']},
        'outcomes': {
            label: {kind: sum(row['outcome'] == kind for row in rows)
                    for kind in ('success', 'collision', 'timeout')}
            for label, rows in (('main', main_rows), ('codex', codex_rows))},
        'benchmarks': make_records('main', main_rows) +
                      make_records('codex', codex_rows),
    }
    args.output.write_text(json.dumps(document, indent=2) + '\n')
    print(json.dumps(document, indent=2))


if __name__ == '__main__':
    main()
