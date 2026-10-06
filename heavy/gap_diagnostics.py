"""MAIN score components for an already selected GAP: explanatory output only."""
import math
import numpy as np
from utils import wrap


def compute_legacy_gap_metrics(gap, boat_pos, boat_heading, target, obstacles, params=None):
    if gap is None:
        return None
    params=params or {}
    c1,c2,point = gap['c1'],gap['c2'],gap['pos']
    bx,by=boat_pos
    rel=point-boat_pos
    distance=math.hypot(*rel)+1e-6
    goal_angle=math.atan2(target[1]-by,target[0]-bx)
    point_angle=math.atan2(rel[1],rel[0])
    heading_align=math.exp(-(wrap(point_angle-goal_angle)/.9)**2)
    head_score=math.exp(-(wrap(point_angle-boat_heading)/.8)**2)
    forward=max((rel[0]*math.cos(goal_angle)+rel[1]*math.sin(goal_angle))/distance,0.)**1.5
    ang1=wrap(math.atan2(c1[1]-by,c1[0]-bx)-boat_heading)
    ang2=wrap(math.atan2(c2[1]-by,c2[0]-bx)-boat_heading)
    lateral=min(max(abs(ang2-ang1)/(np.pi/2),0.),1.)**2
    sym=min(max(1.-abs(abs(ang1)-abs(ang2))/(np.pi/2),0.),1.)
    lateral_full=.6*lateral+.4*sym
    axis=c2-c1
    gap_width=math.hypot(*axis)
    unit=axis/(gap_width+1e-6)
    normal=np.array([-unit[1],unit[0]])
    heading_vec=np.array([math.cos(boat_heading),math.sin(boat_heading)],dtype=np.float32)
    width=min(abs(float(normal@heading_vec)),abs(float(normal@(rel/distance))))
    clear=1.;min_clear=9999.
    obs=np.asarray(obstacles)
    if len(obs):
        d2=(obs[:,0]-bx)**2+(obs[:,1]-by)**2
        path_obs=obs[(d2<=(distance+200.)**2) &
                     (np.sum((obs[:,:2]-c1)**2,axis=1)>28.**2) &
                     (np.sum((obs[:,:2]-c2)**2,axis=1)>28.**2)]
        if len(path_obs):
            t=np.clip((path_obs[:,:2]-boat_pos)@rel/(distance*distance),0.,1.)
            min_clear=max(0.,float(np.min(np.linalg.norm(path_obs[:,:2]-
                (boat_pos+t[:,None]*rel),axis=1)-path_obs[:,2])))
            a,b=c1-boat_pos,c2-boat_pos
            area=abs(a[0]*b[1]-b[0]*a[1])
            if area>1.:
                v=path_obs[:,:2]-boat_pos
                aa,ab,bb=float(a@a),float(a@b),float(b@b)
                va,vb=v@a,v@b
                inv=1./(aa*bb-ab*ab+1e-12)
                u=(bb*va-ab*vb)*inv;w=(aa*vb-ab*va)*inv
                boundary=np.minimum(np.minimum(u,w),1.-u-w)
                near=boundary>-.15
                m=boundary[near]
                density=np.sum(np.where(m>=0.,1.,np.exp(-(m/.08)**2))*
                               np.clip(path_obs[near,2]/17.,.5,2.))
                clear=math.exp(-float(density)/1.5)
    weights=dict(Align=params.get('align_exp',6.),
        Heading=params.get('heading_exp',params.get('boat_align_exp',params.get('head_exp',4.))),
        Forward=params.get('fwd_exp',6.),Width=params.get('width_exp',8.),
        Clear=params.get('clear_exp',3.),Perpend=params.get('perp_exp',2.))
    raw=dict(Align=heading_align,Heading=head_score,Forward=forward,
             Width=width,Clear=clear,Perpend=abs(float(unit[1])))
    bases=dict(raw,Heading=max(head_score,.05),Perpend=max(raw['Perpend'],.05))
    # Log form preserves MAIN's product but keeps diagnostic-only extreme
    # exponents finite. No diagnostic threshold can invalidate a selected GAP.
    terms=[(weights[k],bases[k]) for k in weights]+[(.5,lateral_full),(.2,min(gap_width/90.,1.))]
    if any(value<=0. and weight>0. for weight,value in terms):
        score=0.
    else:
        exponent=sum(float(weight)*math.log(max(float(value),1e-300)) for weight,value in terms)
        score=math.exp(min(700.,max(-745.,exponent)))
    return dict(score=score,min_clear=min_clear,
        factors={k:dict(raw=float(raw[k]),w=float(weights[k])) for k in weights},
        diagnostics_only=True)
