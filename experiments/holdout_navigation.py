"""One frozen paired holdout. Do not tune on the output of this script."""
import json
from pathlib import Path
from experiments.compare_navigation import run
from experiments.sampling_navigation import SamplingConfig


def main():
    root=Path('data/architecture_rethink/holdout24')
    if root.exists():
        raise SystemExit('Holdout exists; preserve it rather than rerunning')
    root.mkdir(parents=True)
    cfg=SamplingConfig(smooth_weight=3.,yaw_command_step=.1)
    results=[]
    for seed in range(2200,2224):
        for mode in ['baseline','corridor']:
            result=run(seed,mode,cfg,root/mode)
            results.append(result)
            (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':main()
