"""Small deterministic one-parameter search on the declared tuning split only."""
from dataclasses import replace
import json
from pathlib import Path
from experiments.compare_navigation import run
from experiments.sampling_navigation import SamplingConfig


def main():
    root=Path('data/architecture_rethink/smoothness_search')
    if root.exists():
        raise SystemExit('Search output exists; do not overwrite experiments')
    root.mkdir(parents=True)
    trials=[]
    for mode,weight in [(m,w) for m in ['corridor','direct'] for w in [10.,30.]]:
        cfg=replace(SamplingConfig(),smooth_weight=weight)
        rows=[run(seed,mode,cfg,root/f'{mode}_weight_{weight:g}') for seed in [2000,2081,2189]]
        # Lexicographic safety first, then weighted efficiency/naturalness score.
        # Fixed budgets; do not expand the search onto validation/holdout maps.
        failures=sum(r['outcome']!='success' for r in rows)
        collisions=sum(r['outcome'] in ('collision','boundary') for r in rows)
        mean=lambda key:sum(r[key] for r in rows)/len(rows)
        score=10000*collisions+1000*failures+mean('time_s')+2*mean('yaw_reversals')+20*mean('steering_tv_per_s')+.2*mean('peak_yaw_accel_deg_s2')+.1*sum(r['planning_tick']['p99_ms'] for r in rows)/len(rows)
        trials.append(dict(mode=mode,weight=weight,score=score,collisions=collisions,failures=failures,
                           mean_time_s=mean('time_s'),mean_reversals=mean('yaw_reversals'),
                           mean_peak_accel=mean('peak_yaw_accel_deg_s2'),mean_tv=mean('steering_tv_per_s')))
        (root/'summary.json').write_text(json.dumps(trials,indent=2)+'\n')
    print(json.dumps(trials,indent=2))


if __name__=='__main__':main()
