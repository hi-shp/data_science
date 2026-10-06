"""Persistent, display-only passage identities; control never reads this state."""
import math
import numpy as np
from utils import wrap
from heavy.gap_annotation import (gap_identity,nearly_same_gate,
                                  select_route_gaps,select_second_gap,
                                  second_gap_band,second_gap_separated,
                                  crossing_in_front,forward_crossings,segments_intersect,
                                  local_passage_groups)

MAIN_WAYPOINT_COMPLETION_RADIUS = 60.


def waypoint_completion_reason(gap,boat,heading):
    """MAIN main.py's completion predicates, applied to the dynamic waypoint.

    Coordinates and all boundaries are the existing MAIN pixel conventions;
    the waypoint remains the route intersection, not the obstacle midpoint.
    """
    if gap is None:
        return None
    point=gap['pos'];dx=point[0]-boat[0];dy=point[1]-boat[1]
    distance=math.hypot(dx,dy)
    if distance<MAIN_WAYPOINT_COMPLETION_RADIUS:
        return 'proximity'
    c1,c2=gap.get('c1'),gap.get('c2')
    if c1 is not None and c2 is not None:
        gx=c2[0]-c1[0];gy=c2[1]-c1[1];length=math.hypot(gx,gy)
        if length>1e-3:
            ux=gx/length;uy=gy/length;nx=-uy;ny=ux
            if nx*math.cos(heading)+ny*math.sin(heading)<0:
                nx=-nx;ny=-ny
            rx=boat[0]-point[0];ry=boat[1]-point[1]
            normal=rx*nx+ry*ny;lateral=abs(rx*ux+ry*uy)
            if 15.0<=normal<60.0 and lateral<length/2.0+20.0:
                return 'gate_passed'
    if abs(wrap(math.atan2(dy,dx)-heading))>1.6580627893946132 and distance<75:
        return 'behind_nearby'
    return None


def crossed_portal(previous,current,gap):
    """Actual finite boat-motion/portal crossing, not waypoint proximity."""
    if previous is None or gap is None:
        return False
    start,end=gap['c1'],gap['c2'];axis=end-start;motion=current-previous
    cross=lambda a,b:float(a[0]*b[1]-a[1]*b[0])
    den=cross(motion,axis)
    if abs(den)<1e-10:
        return False
    t=cross(start-previous,axis)/den
    s=cross(start-previous,motion)/den
    # Count arrival exactly on the gate, but not departing from it later.
    return 1e-8<t<=1.+1e-8 and 0.<=s<=1.


class GapAnnotationState:
    def __init__(self,completion_hull=None):
        # Optional presentation latch; never a controller input. Geometry is
        # the existing physical hull in the same pixel units as MAIN's zone.
        self.completion_hull=(None if completion_hull is None else
                              np.asarray(completion_hull).reshape(-1,2))
        self.reset()

    def reset(self):
        self.current_first_gap=self.current_second_gap=None
        self.current_first_crossing=self.current_second_crossing=None
        self.first_gap_selected_frame=self.second_gap_selected_frame=-1
        self.first_gap_last_valid_frame=self.second_gap_last_valid_frame=-1
        self.first_gap_switch_count=self.second_gap_switch_count=0
        self.previous_boat=None
        self.passed_annotations=[]
        self.completed_identities=set()
        self.first_latched=False
        self.switch_reason={'first':'reset','second':'reset'}
        self.second_preferred_band=None
        self.last_gap={'first':None,'second':None}
        self.pending={'first':None,'second':None}
        self.last_identity={'first':None,'second':None}
        self.identity_switches={'first':0,'second':0}
        self.lifetimes={'first':[],'second':[]}
        self.selection_events=[]

    def completion_reason(self,gap,boat,heading):
        reason=waypoint_completion_reason(gap,boat,heading)
        if gap is None or self.completion_hull is None:
            return reason
        # A physical finite gate crossing must release even before MAIN's
        # downstream normal-distance test, and even if no render latched yet.
        if crossed_portal(self.previous_boat,np.asarray(boat),gap):
            return 'gate_crossed'
        if reason!='proximity':
            return reason
        delta=gap['pos']-boat;distance=float(np.linalg.norm(delta))
        if distance==0. or not crossing_in_front(gap,boat,heading):
            return reason
        # Inside MAIN's existing completion zone, let the hull approach the
        # intersection before releasing it. No hold timer: at higher speed
        # this geometry naturally clears sooner. Never wait for exact center.
        c,s=math.cos(heading),math.sin(heading)
        direction=np.array([c*delta[0]+s*delta[1],-s*delta[0]+c*delta[1]])/distance
        reach=min(MAIN_WAYPOINT_COMPLETION_RADIUS,
                  float(np.max(self.completion_hull@direction)))
        return reason if distance<=reach else None

    def latch_first(self,gap):
        if gap is not None and gap_identity(gap)==gap_identity(self.current_first_gap):
            self.first_latched=True

    def _commit(self,slot,gap,frame,reason):
        old=getattr(self,'current_'+slot+'_gap')
        changed=gap_identity(old)!=gap_identity(gap)
        if changed:
            if slot=='first':
                self.first_latched=False
            if old is not None:
                self.lifetimes[slot].append(max(0,frame-getattr(self,slot+'_gap_selected_frame')))
            self.selection_events.append(dict(frame=frame,slot=slot,
                previous=gap_identity(old),current=gap_identity(gap),reason=reason))
            if gap is not None:
                identity=gap_identity(gap)
                if self.last_identity[slot] is not None and self.last_identity[slot]!=identity:
                    self.identity_switches[slot]+=1
                self.last_identity[slot]=identity
            # Count presentation changes between two visible identities;
            # acquisition/removal are separately identified by switch_reason.
            if old is not None and gap is not None:
                key=slot+'_gap_switch_count'
                setattr(self,key,getattr(self,key)+1)
            setattr(self,slot+'_gap_selected_frame',frame if gap is not None else -1)
        setattr(self,'current_'+slot+'_gap',None if gap is None else gap.copy())
        setattr(self,'current_'+slot+'_crossing',None if gap is None else gap['pos'].copy())
        if gap is not None:
            self.last_gap[slot]=gap.copy()
            setattr(self,slot+'_gap_last_valid_frame',frame)
        self.switch_reason[slot]=reason

    def _acquire(self,slot,choose,validate,generation,immediate,eligible):
        """One complete renewal for first; two for the more distant second.

        Invalid incumbent/pending markers are never drawn. A pending pair is
        validated before any ranking, so a slightly prettier newcomer cannot
        keep resetting confirmation. No wall-clock/recovery timer is involved.
        """
        required=2 if slot=='first' else 3
        pending=self.pending[slot]
        if pending is not None:
            gap,_=validate(pending['gap'])
            if gap is not None and eligible(gap):
                if generation!=pending['generation']:
                    pending['count']+=1;pending['generation']=generation
                pending['gap']=gap
                if immediate or pending['count']>=required:
                    self.pending[slot]=None
                    return gap,'confirmed'
                return None,'confirmation_pending'
            self.pending[slot]=None
        gap=choose()
        if gap is None:
            return None,'no_valid_crossing'
        if immediate:
            return gap,'acquired'
        self.pending[slot]=dict(gap=gap,count=1,generation=generation)
        return None,'confirmation_pending'

    def update(self,events,path,headings,hull,speed,response,margin,profile,boat,frame,validate,
               heading=None,generation=None):
        generation=frame if generation is None else generation
        if heading is not None:
            events=forward_crossings(events,boat,heading)
        def valid_current(old):
            if gap_identity(old) in self.completed_identities:
                return None,'visited'
            result=validate(old)
            if isinstance(result,tuple):
                gap,reason=result
            else:
                gap=result;reason='route_or_safety_invalid' if gap is None else 'valid'
            if gap is not None and heading is not None and not crossing_in_front(gap,boat,heading):
                return None,'behind_bow'
            return gap,reason
        # A temporarily absent annotation can resume its SAME current pair
        # without reranking. The remembered segment is never a drawing fallback:
        # validate must supply a currently existing, exact-safe crossing.
        old_first=self.current_first_gap or self.last_gap['first']
        old_second=self.current_second_gap or self.last_gap['second']
        completion=self.completion_reason(old_first,boat,0. if heading is None else heading)
        passed=completion is not None
        self.passed_annotations=[g for g in self.passed_annotations
            if np.linalg.norm(boat-g['pos'])<=g['passage_extent']]
        if passed:
            self.completed_identities.add(gap_identity(old_first))
            self.passed_annotations.append(old_first)
            self.last_gap['first']=None;self.pending['first']=None
        events=[group for group in events if not any(
            nearly_same_gate(member,old,margin)
            for member in group.get('presentation_candidates',(group,))
            for old in self.passed_annotations)]
        # A visited pair cannot return, but must not hide an independent,
        # unvisited pair that happens to share a future presentation group.
        if self.completed_identities:
            events=local_passage_groups([member for group in events
                for member in group.get('presentation_candidates',(group,))
                if gap_identity(member) not in self.completed_identities])
        first,reason=(valid_current(old_first) if old_first is not None and not passed
                      else (None,'completed:'+completion if passed else 'no_previous_gap'))
        if first is not None:
            reason='retained' if self.current_first_gap is not None else 'same_pair_resumed'
            self.pending['first']=None
        if first is None and passed and old_second is not None:
            first,_=valid_current(old_second)
            if first is not None:
                reason='promoted_second';self.pending['first']=None
        if first is None:
            invalid_reason=reason
            first,status=self._acquire('first',lambda:select_route_gaps(
                events,path,headings,hull,speed,response,margin,profile)[0],
                valid_current,generation,passed or self.last_identity['first'] is None,lambda g:True)
            reason=invalid_reason+':'+status
        second=None
        second_reason='no_previous_gap'
        self.second_preferred_band=(None if first is None else second_gap_band(
            events,first,path,headings,hull,speed,response,margin,profile))
        def second_eligible(gap):
            return (first is not None and second_gap_separated(first,gap,
                    self.second_preferred_band,retained=True) and not segments_intersect(first,gap))
        if first is not None and old_second is not None:
            candidate,second_reason=valid_current(old_second)
            if candidate is not None:
                if second_eligible(candidate):
                    second=candidate
                    second_reason='retained' if self.current_second_gap is not None else 'same_pair_resumed'
                    self.pending['second']=None
                else:
                    second_reason='first_structure_or_separation_or_segment_crossing'
        if second is None and first is not None:
            invalid_reason=second_reason
            second,status=self._acquire('second',lambda:select_second_gap(
                events,first,path,headings,hull,speed,response,margin,profile),
                valid_current,generation,passed or self.last_identity['second'] is None,second_eligible)
            second_reason=invalid_reason+':'+status
        self._commit('first',first,frame,reason)
        self._commit('second',second,frame,second_reason)
        self.previous_boat=np.asarray(boat).copy()
        return first,second

    def apply_visible(self,first,second,frame):
        # Post-goal markers must not survive in the persistent UI state.
        for slot,gap in (('first',first),('second',second)):
            if gap is None and getattr(self,'current_'+slot+'_gap') is not None:
                self._commit(slot,None,frame,'goal_clip')
                self.last_gap[slot]=None;self.pending[slot]=None

    def diagnostics(self,frame,dt=.04):
        def average(slot):
            values=self.lifetimes[slot]
            active=getattr(self,'current_'+slot+'_gap') is not None
            total=sum(values)+(frame-getattr(self,slot+'_gap_selected_frame') if active else 0)
            return total*dt/max(1,len(values)+int(active))
        return dict(first_gap_identity_switches_including_hidden=self.identity_switches['first'],
                    first_waypoint_latched=self.first_latched,
                    second_gap_identity_switches_including_hidden=self.identity_switches['second'],
                    average_first_gap_lifetime_sim_s=average('first'),
                    average_second_gap_lifetime_sim_s=average('second'),
                    pending_first_count=0 if self.pending['first'] is None else self.pending['first']['count'],
                    pending_second_count=0 if self.pending['second'] is None else self.pending['second']['count'],first_gap_switch_count=self.first_gap_switch_count,
                    second_gap_switch_count=self.second_gap_switch_count,
                    first_gap_age=0 if self.current_first_gap is None else frame-self.first_gap_selected_frame,
                    second_gap_age=0 if self.current_second_gap is None else frame-self.second_gap_selected_frame,
                    switch_reason=self.switch_reason.copy(),
                    second_preferred_band=self.second_preferred_band)
