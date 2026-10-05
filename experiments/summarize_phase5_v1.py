"""Read the five stored motion comparisons; no simulation is executed."""
import json
import math
import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import MAP_W, SIM_H
DATA = ROOT/'data/main_heavy'
GOAL = np.array([MAP_W-100., SIM_H/2.])/50.
SEEDS = (2000,2069,2081,2003,2004)


def metrics(steps, obstacles):
    x=np.array([s['x'] for s in steps])/50.
    y=np.array([s['y'] for s in steps])/50.
    heading=np.array([s['heading'] for s in steps])
    yaw=np.array([s['yaw'] for s in steps])
    command=np.array([s['steer'] for s in steps])*.5
    speed=np.array([s['speed'] for s in steps])
    surge=np.array([s['surge'] for s in steps])
    time=np.array([s['t'] for s in steps])
    dt=.04
    error=np.arctan2(np.sin(np.arctan2(GOAL[1]-y,GOAL[0]-x)-heading),
                     np.cos(np.arctan2(GOAL[1]-y,GOAL[0]-x)-heading))
    # The same offline straight-window definition for both controllers.
    straight=(abs(error)<.15)&(speed>.4)
    for ox,oy,r in obstacles:
        phase=time+.05*(ox+oy)
        dx=(ox+np.sin(phase)*r*.2)/50.-x
        dy=(oy+np.cos(phase*1.2)*r*.2)/50.-y
        ahead=dx*np.cos(heading)+dy*np.sin(heading)
        side=abs(-dx*np.sin(heading)+dy*np.cos(heading))
        straight &= ~((ahead>0.)&(ahead<3.+r/50.)&(side<.7+r/50.))
    contiguous=straight[1:]&straight[:-1]
    def reversals(values,mask=None):
        sign=np.where(values>.02,1,np.where(values<-.02,-1,0))
        count=0;last=0
        for i,v in enumerate(sign):
            if mask is not None and not mask[i]:last=0;continue
            if v:
                count+=int(last!=0 and last!=v);last=v
        return count
    reverse=surge<-.05
    onset=reverse&~np.r_[False,reverse[:-1]]
    length=np.hypot(np.diff(x),np.diff(y))
    return dict(completion_s=float(time[-1]),path_length_m=float(length.sum()),
        command_changes_per_s=float((abs(np.diff(command))>.001).sum()/time[-1]),
        command_tv=float(abs(np.diff(command)).sum()),
        command_reversals=reversals(command),
        straight_reversals=reversals(command,straight),
        straight_command_changes_per_s=float(((abs(np.diff(command))>.001)&contiguous).sum()/max(straight.sum()*dt,dt)),
        straight_command_tv=float(abs(np.diff(command))[contiguous].sum()),
        straight_yaw_rms=float(np.sqrt(np.mean(yaw[straight]**2))) if straight.any() else None,
        straight_heading_error_rms=float(np.sqrt(np.mean(error[straight]**2))) if straight.any() else None,
        straight_lateral_rms_m=float(np.sqrt(np.mean((y[straight]-GOAL[1])**2))) if straight.any() else None,
        mean_abs_yaw=float(abs(yaw).mean()),peak_yaw=float(abs(yaw).max()),
        reverse_count=int(onset.sum()),reverse_duration_s=float(reverse.sum()*dt),
        reverse_distance_m=float((-np.minimum(surge,0.)).sum()*dt),
        minimum_hull_clearance_m=float(min(s['actual_clearance_m'] for s in steps)))


def main():
    rows=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(5,4,figsize=(18,17))
    for n,seed in enumerate(SEEDS):
        heavy=json.loads((DATA/f'v1_inspection_{seed}.trace.json').read_text())
        summary=json.loads((DATA/f'v1_inspection_{seed}.jsonl').read_text())
        codex=json.loads((DATA/f'v1_codex_{seed}.trace.json').read_text())
        assert summary['map_hash']==codex['map_hash'],seed
        row=dict(seed=seed,map_hash=summary['map_hash'],heavy_outcome=summary['outcome'],codex_outcome=codex['outcome'],
                 main_heavy_v1=metrics(heavy['steps'],heavy['obstacles']),codex=metrics(codex['steps'],codex['obstacles']))
        rows.append(row)
        for data,color,label in ((heavy,'tab:blue','MAIN_HEAVY V1'),(codex,'tab:orange','CODEX')):
            steps=data['steps'];t=[s['t'] for s in steps]
            axes[n,0].plot([s['x']/50. for s in steps],[s['y']/50. for s in steps],color=color,label=label)
            axes[n,1].plot(t,[s['heading'] for s in steps],color=color)
            axes[n,2].plot(t,[s['steer']*.5 for s in steps],color=color)
            axes[n,3].plot(t,[s['speed'] for s in steps],color=color)
        axes[n,0].set_title(f'Seed {seed}: vessel trajectory',fontsize=13)
        axes[n,0].set_xlabel('x (m)',fontsize=11);axes[n,0].set_ylabel('y (m)',fontsize=11)
        axes[n,0].legend(fontsize=10)
        for i,title in enumerate(('Heading (rad)','Yaw command (rad/s)','Speed (m/s)'),1):
            axes[n,i].set_title(title,fontsize=13);axes[n,i].set_xlabel('Simulation time (s)',fontsize=11)
        for ax in axes[n]:ax.tick_params(labelsize=10);ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(DATA/'v1_codex_motion_comparison.png',dpi=120);plt.close(fig)
    (DATA/'v1_codex_motion_comparison.json').write_text(json.dumps(rows,indent=2))
    print(json.dumps([dict(seed=r['seed'],heavy_reverse=r['main_heavy_v1']['reverse_duration_s'],codex_reverse=r['codex']['reverse_duration_s'],heavy_straight_reversal=r['main_heavy_v1']['straight_reversals'],codex_straight_reversal=r['codex']['straight_reversals']) for r in rows]))

if __name__=='__main__':main()
