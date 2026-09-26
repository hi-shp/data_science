"""Plot recorded evidence, never conceptual or fabricated vessel motion."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--baseline',type=Path,required=True)
    p.add_argument('--candidate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[2000,2081,2189])
    args=p.parse_args()
    plt.rcParams.update({'font.size':10,'axes.labelsize':11,'axes.titlesize':13,'legend.fontsize':10})
    fig,axes=plt.subplots(4,len(args.seeds),figsize=(6*len(args.seeds),14),constrained_layout=True,squeeze=False)
    for j,seed in enumerate(args.seeds):
        obs=np.array(json.loads((args.baseline/f'{seed}_map.json').read_text()))/50.
        for x,y,r in obs:
            axes[0,j].add_patch(plt.Circle((x,y),r,color='0.65'))
        for folder,label,color in [(args.baseline,'Baseline','#285f9e'),(args.candidate,'Candidate','#cf5900')]:
            trace=np.genfromtxt(folder/f'{seed}_trace.csv',delimiter=',',names=True)
            result=json.loads((folder/f'{seed}.json').read_text())
            axes[0,j].plot(trace['x_m'],trace['y_m'],color=color,label=f"{label}: {result['outcome']}, {result['time_s']:.2f} s",lw=1.5)
            axes[1,j].plot(trace['time_s'],np.rad2deg(np.unwrap(trace['heading_rad'])),color=color,lw=1.2)
            axes[2,j].plot(trace['time_s'],trace['command_yaw_rate']/.5,color=color,lw=1.)
            axes[3,j].plot(trace['time_s'],np.hypot(trace['u_m_s'],trace['v_m_s']),color=color,lw=1.2)
        axes[0,j].set(xlabel='x (m)',ylabel='y (m)',xlim=(0,36),ylim=(0,12.6),title=f'Seed {seed}: recorded trajectory')
        axes[0,j].set_aspect('equal');axes[0,j].legend(loc='upper center',bbox_to_anchor=(.5,1.38),frameon=False)
        for i,label in [(1,'Heading (deg, unwrapped)'),(2,'Yaw command / 0.5 rad/s'),(3,'Linear speed (m/s)')]:
            axes[i,j].set(xlabel='Simulation time (s)',ylabel=label)
            axes[i,j].grid(alpha=.2)
    fig.suptitle('Identical maps and frozen vessel dynamics; steering is a command, not a measured rudder angle',fontsize=14)
    fig.savefig(args.output,dpi=140)
    plt.close(fig)


if __name__=='__main__':main()
