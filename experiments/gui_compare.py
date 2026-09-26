"""Run the existing real-display profiler with an opt-in navigation prototype."""
import argparse
import importlib.util
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--mode',choices=['baseline','corridor','direct'],required=True)
    parser.add_argument('--weight',type=float,default=3.)
    parser.add_argument('--config',type=Path)
    parser.add_argument('--speed',type=int,default=1)
    parser.add_argument('--seed',type=int,default=2081)
    parser.add_argument('--duration',type=float,default=5.)
    parser.add_argument('--max-steps',type=int,default=10000)
    parser.add_argument('--label',required=True)
    args=parser.parse_args()
    configuration=json.loads(args.config.read_text()) if args.config else {'smooth_weight':args.weight}
    # Importing the headless comparison helper sets SDL defaults; restore the
    # real display selection before constructing the GUI environment.
    driver=os.environ.get('SDL_VIDEODRIVER')
    from experiments.compare_navigation import experimental_step
    from experiments.sampling_navigation import SamplingNavigator,SamplingConfig
    if args.mode!='baseline':
        # Compile outside the timed/rendered loop; the kernel has the same
        # specialization for every physical parameter value.
        import numpy as np
        from vessel_dynamics import VesselParameters
        from experiments.fast_rollout import compiled_rollout,parameter_vector
        if compiled_rollout is not None:
            compiled_rollout(np.zeros(8),np.zeros((1,1,2)),np.empty((0,3)),
                             36.,12.6,.04,parameter_vector(VesselParameters()))
    if driver is None:
        os.environ.pop('SDL_VIDEODRIVER',None)
    else:
        os.environ['SDL_VIDEODRIVER']=driver
    spec=importlib.util.spec_from_file_location('gui_probe',Path('data/realtime_perf/gui_probe.py'))
    probe=importlib.util.module_from_spec(spec);spec.loader.exec_module(probe)
    if args.mode!='baseline':
        def advance(env,step_idx=0,sub_steps=1):
            if env.frame==0 or not hasattr(env,'experimental_navigator'):
                env.experimental_navigator=SamplingNavigator(env.dynamics,env.dt,args.mode,SamplingConfig(**configuration))
            return experimental_step(env,env.experimental_navigator)
        probe.main.advance=advance
    probe.run(args.speed,duration_s=args.duration,max_steps=args.max_steps,
              seed=args.seed,label=args.label)


if __name__=='__main__':main()
