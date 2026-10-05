"""Persistent, display-only passage identities; control never reads this state."""
import numpy as np
from heavy_gap_annotation import (gap_identity,nearly_same_gate,
                                  select_route_gaps,select_second_gap,
                                  second_gap_band,second_gap_separated,
                                  crossing_in_front,forward_crossings,segments_intersect)


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
    def __init__(self):
        self.reset()

    def reset(self):
        self.current_first_gap=self.current_second_gap=None
        self.current_first_crossing=self.current_second_crossing=None
        self.first_gap_selected_frame=self.second_gap_selected_frame=-1
        self.first_gap_last_valid_frame=self.second_gap_last_valid_frame=-1
        self.first_gap_switch_count=self.second_gap_switch_count=0
        self.previous_boat=None
        self.passed_annotations=[]
        self.switch_reason={'first':'reset','second':'reset'}
        self.second_preferred_band=None
        self.last_gap={'first':None,'second':None}
        self.pending={'first':None,'second':None}
        self.last_identity={'first':None,'second':None}
        self.identity_switches={'first':0,'second':0}
        self.lifetimes={'first':[],'second':[]}
        self.selection_events=[]

    def _commit(self,slot,gap,frame,reason):
        old=getattr(self,'current_'+slot+'_gap')
        changed=gap_identity(old)!=gap_identity(gap)
        if changed:
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
        passed=crossed_portal(self.previous_boat,boat,old_first)
        second_passed=crossed_portal(self.previous_boat,boat,old_second)
        self.passed_annotations=[g for g in self.passed_annotations
            if np.linalg.norm(boat-g['pos'])<=g['passage_extent']]
        for slot,gap,was_passed in (('first',old_first,passed),('second',old_second,second_passed)):
            if was_passed:
                if not any(gap_identity(g)==gap_identity(gap) for g in self.passed_annotations):
                    self.passed_annotations.append(gap)
                self.last_gap[slot]=None;self.pending[slot]=None
        events=[group for group in events if not any(
            nearly_same_gate(member,old,margin)
            for member in group.get('presentation_candidates',(group,))
            for old in self.passed_annotations)]
        first,reason=(valid_current(old_first) if old_first is not None and not passed
                      else (None,'passed' if passed else 'no_previous_gap'))
        if first is not None:
            reason='retained' if self.current_first_gap is not None else 'same_pair_resumed'
            self.pending['first']=None
        if first is None and passed and old_second is not None and not second_passed:
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
        second_reason='passed' if second_passed else 'no_previous_gap'
        self.second_preferred_band=(None if first is None else second_gap_band(
            events,first,path,headings,hull,speed,response,margin,profile))
        def second_eligible(gap):
            return (first is not None and second_gap_separated(first,gap,
                    self.second_preferred_band,retained=True) and not segments_intersect(first,gap))
        if first is not None and old_second is not None and not second_passed:
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
