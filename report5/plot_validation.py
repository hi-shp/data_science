"""Create figures only from the recorded benchmark CSV/JSON files."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

ROOT = Path(__file__).resolve().parent
plt.rcParams.update({'font.size':11,'axes.labelsize':11,'axes.titlesize':13,
                     'xtick.labelsize':11,'ytick.labelsize':11,'legend.fontsize':10})
LABELS = ['Original','Inertia x2 + predictive control']
COLORS = ['#bb542d','#176b99']


def read(name):
    root = ROOT/name
    summary = json.loads((root/'summary.json').read_text())
    with (root/'episodes.csv').open() as f:
        episodes = list(csv.DictReader(f))
    with (root/'trajectories.csv').open() as f:
        traces = list(csv.DictReader(f))
    return summary,episodes,traces


def main():
    data = [read('original_navigation'),read('inertia_navigation')]
    summaries = [d[0] for d in data]
    for a,b in zip(data[0][1],data[1][1]):
        assert a['seed']==b['seed'] and a['map_hash']==b['map_hash'], 'Unpaired maps'
    fields = [('Success rate (%)','success'),('Successful arrival time (s)','mean_success_time_s'),
              ('Mean peak yaw rate (deg/s)','mean_peak_yaw_deg_s'),
              ('Mean peak yaw acceleration (deg/s²)','mean_peak_yaw_accel_deg_s2'),
              ('Yaw reversals (>3 deg/s)','mean_yaw_reversals'),
              ('Normalized command variation / s','mean_steering_tv_per_s')]
    fig,axes = plt.subplots(2,3,figsize=(16,9),layout='constrained')
    for ax,(label,field) in zip(axes.flat,fields):
        values = [100*s['success']/s['episodes'] if field=='success' else s[field] for s in summaries]
        ax.bar([0,1],values,color=COLORS,width=.55)
        ax.set_xticks([0,1],['Original','Inertia x2'])
        ax.set_title(label)
        ax.set_ylim(0,max(values)*1.25)
        if field == 'success':
            ax.set_ylim(0,108)
            ax.set_yticks(np.arange(0,101,20))
        ax.grid(axis='y',alpha=.2)
        ax.set_axisbelow(True)
        for x,value in enumerate(values):
            ax.text(x,value+max(values)*.035,f'{value:.2f}',ha='center',fontsize=12)
    fig.suptitle('Paired simulation validation: 200 maps, seeds 2000–2199',fontsize=16)
    fig.savefig(ROOT/'validation_metrics.png',dpi=180)
    fig.savefig(ROOT/'validation_metrics.svg')
    plt.close(fig)

    seed = 2000  # fixed first validation seed, not selected for a favorable result
    traces=[]
    for _,_,rows in data:
        traces.append({key:np.array([float(row[key]) for row in rows if int(row['seed'])==seed])
                       for key in rows[0] if key!='seed'})
    maps=json.loads((ROOT/'original_navigation'/'maps.json').read_text())
    fig,axes=plt.subplots(3,2,figsize=(16,12),layout='constrained')
    for ax,trace,color,label in zip(axes[0],traces,COLORS,LABELS):
        for x,y,r in maps[str(seed)]:
            ax.add_patch(Circle((x/50,y/50),r/50,color='#939da6',alpha=.8))
        ax.plot(trace['x_m'],trace['y_m'],color=color,lw=2)
        for i in np.linspace(0,len(trace['time_s'])-1,9).astype(int):
            h=trace['heading_rad'][i];c,s=np.cos(h),np.sin(h)
            boat=np.array([[.84,0],[-.6,.4],[-.84,0],[-.6,-.4]])
            rotated=boat@np.array([[c,s],[-s,c]])+np.array([trace['x_m'][i],trace['y_m'][i]])
            ax.add_patch(Polygon(rotated,facecolor=color,alpha=.4))
        ax.set_xlim(0,36);ax.set_ylim(12.6,0);ax.set_aspect('equal')
        ax.set_xlabel('East (m)');ax.set_ylabel('Map Y (m, down)');ax.set_title(label)
    for trace,color,label in zip(traces,COLORS,LABELS):
        t=trace['time_s']
        axes[1,0].plot(t,np.rad2deg(np.unwrap(trace['heading_rad'])),color=color,label=label)
        axes[1,1].plot(t,np.rad2deg(trace['yaw_rate_rad_s']),color=color,label=label)
        axes[2,0].plot(t,trace['steering_command'],color=color,label=label)
        axes[2,1].plot(t,trace['speed_m_s'],color=color,label=label)
    for ax,title,ylabel in zip(axes[1:].flat,
                              ['Heading','Yaw rate','Control command (not rudder angle)','Speed'],
                              ['Heading (deg, east = 0)','Yaw rate (deg/s)','Normalized command','Speed (m/s)']):
        ax.set_title(title);ax.set_xlabel('Simulation time (s)');ax.set_ylabel(ylabel)
        ax.grid(alpha=.2);ax.legend(loc='best')
    fig.suptitle('Recorded trajectory and motion — fixed validation seed 2000',fontsize=16)
    fig.savefig(ROOT/'trajectory_and_motion.png',dpi=180)
    fig.savefig(ROOT/'trajectory_and_motion.svg')
    plt.close(fig)


if __name__=='__main__':main()
