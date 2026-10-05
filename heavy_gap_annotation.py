"""Route-first GAP annotations. These arrays never feed the motion controller."""
import math
import numpy as np
from heavy_motion_core.passage_geometry import (
    prepare_hull_edges, fast_surface_clearances)


def route_bezier(points, headings):
    """Interpolating cubic representation within subpixel distance of the rollout.

    Each original knot is retained. Controls deviate at most 0.25 pixel from
    its linear cubic, so a curve cannot invent a different passage/corridor.
    """
    points = np.asarray(points, dtype=float)
    headings = np.unwrap(np.asarray(headings, dtype=float))
    if len(points) < 2:
        return points.copy(), headings.copy()
    delta = np.diff(points, axis=0)
    tangent = np.vstack((delta[0], (delta[:-1]+delta[1:])/2., delta[-1]))
    linear1, linear2 = points[:-1]+delta/3., points[1:]-delta/3.
    change1, change2 = (tangent[:-1]-delta)/3., (delta-tangent[1:])/3.
    def limited(change):
        return change*np.minimum(1., .25/np.maximum(np.linalg.norm(change,axis=1),1e-12))[:,None]
    p1, p2 = linear1+limited(change1), linear2+limited(change2)
    t = np.arange(4, dtype=float)/4.
    curve = ((1-t)[None,:,None]**3*points[:-1,None]
             +3*((1-t)**2*t)[None,:,None]*p1[:,None]
             +3*((1-t)*t*t)[None,:,None]*p2[:,None]
             +(t**3)[None,:,None]*points[1:,None])
    angles = headings[:-1,None]+t[None,:]*np.diff(headings)[:,None]
    return np.vstack((curve.reshape(-1,2),points[-1])), np.r_[angles.ravel(),headings[-1]]


def route_crossings(path, headings, candidates, obstacles, hull, margin, bounds, pixels_per_m=50.):
    """Safe, finite portal traversals, sorted by physical route arc length.

    `path`, obstacles, hull and margin are all in the same pixel coordinate
    system. Orientation-aware polygon safety is checked at every crossing.
    Collinear touches, endpoint grazes, behind-boat and extended-line hits do
    not qualify. No goal/width/heading ranking weights are used.
    """
    if len(path)<2 or not candidates or not len(obstacles):
        return []
    starts = np.asarray([g['c1'] for g in candidates])
    axes = np.asarray([g['c2']-g['c1'] for g in candidates])
    # A finite gate must overlap the completed route's bounding box.
    lower,upper = path.min(axis=0),path.max(axis=0)
    overlaps = np.all(np.maximum(starts,starts+axes)>=lower,axis=1) & np.all(
                      np.minimum(starts,starts+axes)<=upper,axis=1)
    if not np.any(overlaps):
        return []
    candidates=[candidates[i] for i in np.flatnonzero(overlaps)]
    starts,axes = starts[overlaps],axes[overlaps]
    lengths = np.linalg.norm(axes,axis=1)
    units = axes/np.maximum(lengths[:,None],1e-12)
    # Associate GUI clusters with observed geometry only, never ground truth.
    centers = np.vstack((starts,starts+axes))
    identities = np.argmin(np.sum((centers[:,None]-obstacles[None,:,:2])**2,axis=2),axis=1)
    radii = obstacles[identities,2]+np.linalg.norm(centers-obstacles[identities,:2],axis=1)
    r1,r2 = np.split(radii,2)
    id1,id2 = np.split(identities,2)
    # A gate spanning an intervening observed obstacle is not one passage.
    obstacle_offset = obstacles[None,:,:2]-starts[:,None]
    gate_progress = np.sum(obstacle_offset*units[:,None],axis=2)
    gate_normal = np.abs(obstacle_offset[:,:,0]*units[:,None,1]-
                         obstacle_offset[:,:,1]*units[:,None,0])
    interior = ((gate_progress>r1[:,None])&(gate_progress<lengths[:,None]-r2[:,None])
                &(gate_normal<obstacles[None,:,2]))
    interior[np.arange(len(candidates)),id1]=False
    interior[np.arange(len(candidates)),id2]=False
    open_gate = ~np.any(interior,axis=1)
    offset = path[None]-starts[:,None]
    signed = offset[:,:,0]*units[:,None,1]-offset[:,:,1]*units[:,None,0]
    signed[np.abs(signed)<1e-8] = 0.
    denominator = signed[:,:-1]-signed[:,1:]
    fraction = np.divide(signed[:,:-1],denominator,out=np.full_like(denominator,np.inf),
                         where=np.abs(denominator)>1e-12)
    n = len(path)
    indices = np.arange(n)
    nonzero = signed != 0.
    previous = np.maximum.accumulate(np.where(nonzero,indices,-1),axis=1)
    following = np.minimum.accumulate(np.where(nonzero,indices,n)[:,::-1],axis=1)[:,::-1]
    before = np.take_along_axis(signed,np.maximum(previous,0),axis=1)
    after = np.take_along_axis(signed,np.minimum(following,n-1),axis=1)
    before[previous<0]=0.;after[following>=n]=0.
    valid = ((fraction>=0.)&(fraction<1.)&(before[:,:-1]*after[:,1:]<0.)
             &(lengths[:,None]>1e-8)&open_gate[:,None])
    rows,segments = np.nonzero(valid)
    if not len(rows):
        return []
    f = fraction[rows,segments]
    delta = np.diff(path,axis=0)
    segment_length = np.linalg.norm(delta,axis=1)
    arc = np.r_[0.,np.cumsum(segment_length)]
    along = arc[segments]+f*segment_length[segments]
    points = path[segments]+f[:,None]*delta[segments]
    angles = headings[segments]+f*(headings[segments+1]-headings[segments])
    c,s = np.cos(angles),np.sin(angles)
    vertices = np.asarray(hull).reshape(-1,2)
    world_x = c[:,None]*vertices[:,0]-s[:,None]*vertices[:,1]
    world_y = s[:,None]*vertices[:,0]+c[:,None]*vertices[:,1]
    projected = world_x*units[rows,0,None]+world_y*units[rows,1,None]
    # GUI cluster centroids are surface samples, not observed circle centers.
    # Use the actual observed circle's projected support on this exact segment.
    # Adding an isotropic centroid offset to its radius double-counted inward
    # displacement and made still-safe tracked pairs flicker as clusters moved.
    circle_low=np.sum((obstacles[id1[rows],:2]-starts[rows])*units[rows],axis=1)
    circle_high=np.sum((obstacles[id2[rows],:2]-starts[rows])*units[rows],axis=1)
    low=np.maximum(0.,(circle_low+obstacles[id1[rows],2]+margin-projected.min(1))/lengths[rows])
    high=np.minimum(1.,(circle_high-obstacles[id2[rows],2]-margin-projected.max(1))/lengths[rows])
    portal_s = np.sum((points-starts[rows])*units[rows],axis=1)/lengths[rows]
    direction = delta[segments]/np.maximum(segment_length[segments,None],1e-12)
    hull_along = np.max(np.abs(world_x*direction[:,0,None]+world_y*direction[:,1,None]),axis=1)
    possible = ((low<high)&(portal_s>=low)&(portal_s<=high)&(along>1e-8)&(id1[rows]!=id2[rows]))
    rows,segments,f,along,points,angles,low,high,portal_s,hull_along = (
        value[possible] for value in (rows,segments,f,along,points,angles,low,high,portal_s,hull_along))
    if not len(rows):
        return []
    poses = np.column_stack((points,angles))
    scale = pixels_per_m
    edges,bound,box = prepare_hull_edges(hull/scale)
    poses[:, :2] /= scale
    clearance = fast_surface_clearances(poses,obstacles/scale,
        bounds[0]/scale,bounds[1]/scale,edges,bound,box,True)*scale
    events=[]
    for k in np.flatnonzero(clearance>=margin):
        i = int(rows[k]); direction = delta[segments[k]]/max(segment_length[segments[k]],1e-12)
        g = candidates[i].copy()
        g.update(pos=points[k],portal_s=float(portal_s[k]),interval=(float(low[k]),float(high[k])),
                 route_arc=float(along[k]),route_segment=int(segments[k]),route_fraction=float(f[k]),
                 crossing_clearance=float(clearance[k]),crossing_source='prediction_bezier',
                 obstacle_pair=tuple(sorted((int(id1[i]),int(id2[i])))),
                 gate_turn=float(abs(np.dot(direction,units[i]))),
                 passage_extent=float(max(r1[i],r2[i])+margin+hull_along[k]))
        events.append(g)
    # Simultaneous gates: numerical arc tie, then clearance and tangent change.
    events.sort(key=lambda g:(round(g['route_arc'],8),-g['crossing_clearance'],g['gate_turn'],g['obstacle_pair']))
    return local_passage_groups(events)


def local_passage_groups(events):
    """Retain safe presentation alternatives inside an anchored crossing region.

    A shared observed boundary AND overlapping physical crossing neighborhoods
    are required. Comparing to the first crossing prevents transitive chains
    from merging several successive passages. Repeated crossings of an already
    seen obstacle pair do not become a new passage.
    """
    groups=[];seen=set()
    for event in sorted(events,key=lambda g:(g['route_arc'],g['obstacle_pair'],
                                             tuple(g['c1']),tuple(g['c2']))):
        pair=event['obstacle_pair']
        owner=None
        for group in groups:
            anchor=group[0]
            extent=min(event['passage_extent'],anchor['passage_extent'])
            if (set(pair)&set(anchor['obstacle_pair']) and
                0.<=event['route_arc']-anchor['route_arc']<=extent and
                np.linalg.norm(event['pos']-anchor['pos'])<=extent):
                owner=group
                break
        if pair in seen:
            # Multiple GUI clusters may describe the same observed pair;
            # only its first local traversal can supply alternatives.
            if owner is not None and any(g['obstacle_pair']==pair for g in owner):
                owner.append(event)
            continue
        seen.add(pair)
        if owner is None:
            groups.append([event])
        else:
            owner.append(event)
    result=[]
    for group in groups:
        if len(group)==1:
            result.append(group[0])
        else:
            canonical=group[0].copy()
            canonical.update(presentation_candidates=tuple(group),
                             passage_end_arc=max(g['route_arc'] for g in group))
            result.append(canonical)
    return result


def crossing_in_front(gap,boat,heading):
    """Current crossing bearing, inclusive +/-90 degrees; no midpoint test."""
    if gap is None:
        return False
    dx=float(gap['pos'][0]-boat[0]);dy=float(gap['pos'][1]-boat[1])
    projection=dx*math.cos(heading)+dy*math.sin(heading)
    rounding=8*np.finfo(float).eps*max(1.,abs(dx),abs(dy))
    return projection>=-rounding


def forward_crossings(groups,boat,heading):
    """Filter each actual representative before regrouping local passages."""
    return local_passage_groups([g for group in groups
        for g in group.get('presentation_candidates',(group,))
        if crossing_in_front(g,boat,heading)])


def presentation_key(gap):
    """Midpoint is a local representation reference, never the waypoint."""
    a,b=gap['c1'],gap['c2']
    dx=float(b[0]-a[0]);dy=float(b[1]-a[1]);length=math.hypot(dx,dy)
    offset=math.hypot(float((a[0]+b[0])*.5-gap['pos'][0]),
                      float((a[1]+b[1])*.5-gap['pos'][1]))
    return (offset,length,-abs(dy)/max(length,1e-12),
            -gap.get('crossing_clearance',0.),gap['route_arc'],
            gap['obstacle_pair'],tuple(a),tuple(b))


def select_presentation_gap(candidates):
    """Compare only one already chosen crossing region's valid candidates.

    Within one drawable pixel of the best midpoint distance, compactness wins.
    This is raster resolution, not a physical margin or navigation weight.
    The returned position remains the exact existing route intersection.
    """
    keyed=[(presentation_key(g),g) for g in candidates]
    closest=min(key[0] for key,_ in keyed)
    near=[(key,g) for key,g in keyed if key[0]<=closest+1.]
    return min(near,key=lambda item:(item[0][1],item[0][0],item[0][2:]))[1]


def split_display_route(path, first, second):
    """Insert exact selected intersections into the very array being drawn."""
    events=[g for g in (first,second) if g is not None]
    points=[];first_index=None
    for i in range(len(path)-1):
        points.append(path[i])
        for g in events:
            if g['route_segment']==i:
                if np.linalg.norm(points[-1]-g['pos'])>1e-9:
                    points.append(g['pos'])
                if g is first:
                    first_index=len(points)-1
    if not points or np.linalg.norm(points[-1]-path[-1])>1e-9:
        points.append(path[-1])
    result=np.asarray(points)
    if second is None:
        return result,None
    return result[:first_index+1],result[first_index:]


def segments_intersect(first,second):
    """Finite closed segments only: no near-distance/clearance rejection."""
    a,b,c,d=first['c1'],first['c2'],second['c1'],second['c2']
    cross=lambda u,v:float(u[0]*v[1]-u[1]*v[0])
    ab=b-a;cd=d-c;offset=c-a
    denominator=cross(ab,cd)
    tolerance=8*np.finfo(float).eps*max(1.,float(ab@ab),float(cd@cd))
    if abs(denominator)>tolerance:
        t=cross(offset,cd)/denominator;u=cross(offset,ab)/denominator
        return 0.<=t<=1. and 0.<=u<=1.
    if abs(cross(offset,ab))>tolerance:
        return False
    axis=int(abs(ab[1])>abs(ab[0]))
    left,right=sorted((float(a[axis]),float(b[axis])))
    other_left,other_right=sorted((float(c[axis]),float(d[axis])))
    return max(left,other_left)<=min(right,other_right)


def segments_conflict(first, second, margin):
    """Closed-segment intersection or near overlay, with a physical margin."""
    a,b,c,d=first['c1'],first['c2'],second['c1'],second['c2']
    def cross(u,v):
        return float(u[0]*v[1]-u[1]*v[0])
    ab,cd=b-a,d-c
    den=cross(ab,cd)
    if abs(den)>1e-10:
        t,u=cross(c-a,cd)/den,cross(c-a,ab)/den
        if 0.<=t<=1. and 0.<=u<=1.:
            return True
    def distance(point,start,end):
        axis=end-start
        t=np.clip(float((point-start)@axis)/max(float(axis@axis),1e-12),0.,1.)
        return float(np.linalg.norm(point-(start+t*axis)))
    return min(distance(a,c,d),distance(b,c,d),distance(c,a,b),distance(d,a,b))<=2*margin+1e-8


def select_route_gaps(events,path,headings,hull,speed,response_s,margin,profile=None):
    """Safe route-local presentation; measured shape never affects control."""
    if not events:
        return None,None
    # A newly acquired first should not describe a crossing already inside
    # the current hull footprint. Existing first identities can approach it
    # normally: persistence releases on passage/invalidity, not proximity.
    radius=float(np.linalg.norm(hull.reshape(-1,2),axis=1).max())
    groups=local_passage_groups([g for group in events
        for g in group.get('presentation_candidates',(group,)) if g['route_arc']>=radius])
    if not groups:
        return None,None
    first_index=0
    if profile is not None:
        distance_profile=profile['first_route_distance']
        # Select the crossing region by route location only. Gate width,
        # midpoint offset and orientation cannot choose a remote passage.
        regions=[]
        for index,group in enumerate(groups):
            representatives=group.get('presentation_candidates',(group,))
            ahead=[g['route_arc'] for g in representatives
                   if g['route_arc']>=distance_profile['p25']]
            if ahead:
                distance=min(ahead)
                outside=max(distance_profile['p25']-distance,0.,distance-distance_profile['p75'])
                regions.append(((outside,abs(distance-distance_profile['median']),distance),index))
        if regions:
            first_index=min(regions)[1]
    first_group=groups[first_index]
    representatives=first_group.get('presentation_candidates',(first_group,))
    if profile is not None:
        ahead=[g for g in representatives if g['route_arc']>=profile['first_route_distance']['p25']]
        representatives=ahead or representatives
    first=select_presentation_gap(representatives)
    return first,select_second_gap(groups,first,path,headings,hull,speed,response_s,margin,profile)


def gap_identity(gap):
    """Tracked GUI obstacle pair, independent of the changing route crossing."""
    if gap is None:
        return None
    return tuple(sorted(gap['pair'] if 'pair' in gap else gap['obstacle_pair']))


def nearly_same_gate(first,second,margin):
    """Reject duplicate/near-total overlays, not partial overlaps or X gates."""
    if gap_identity(first)==gap_identity(second):
        return True
    a,b,c,d=first['c1'],first['c2'],second['c1'],second['c2']
    axis=b-a;length=float(np.linalg.norm(axis));other=d-c
    if length<1e-8 or np.linalg.norm(other)<1e-8:
        return True
    unit=axis/length
    normal=lambda p:abs(float((p[0]-a[0])*unit[1]-(p[1]-a[1])*unit[0]))
    if max(normal(c),normal(d))>2*margin:
        return False
    lo,hi=sorted((float((c-a)@unit),float((d-a)@unit)))
    overlap=max(0.,min(length,hi)-max(0.,lo))
    return overlap>=min(length,float(np.linalg.norm(other)))-2*margin


def second_gap_band(groups,first,path,headings,hull,speed,response_s,margin,profile=None):
    """Display-only route distances, derived from hull/MAIN spacing, not horizon rank.

    The hull/passage footprint is the minimum distinct crossing distance. MAIN's
    measured median/IQR supplies the preferred band. Available prediction only
    caps that band; making the horizon longer cannot move its nominal far edge.
    """
    radius=float(np.linalg.norm(hull.reshape(-1,2),axis=1).max())
    _,_,arc=_geometry(path)
    first_end=first['route_arc']
    for group in groups:
        if any(gap_identity(g)==gap_identity(first)
               for g in group.get('presentation_candidates',(group,))):
            first_end=max(first_end,group.get('passage_end_arc',group['route_arc']))
            break
    reference=(dict(p25=first['passage_extent'],median=2*radius,
                    p75=4*radius) if profile is None else profile['second_route_separation'])
    footprint=max(2*margin,radius,min(first['passage_extent'],reference['p25']))
    minimum=first_end-first['route_arc']+footprint
    # Evaluate turn footprint within a fixed local window, not over the whole
    # growing prediction. This is a presentation scale, never a control veto.
    stop=min(float(arc[-1]),first['route_arc']+reference['p75'])
    mask=(arc>first['route_arc'])&(arc<stop)
    angles=np.r_[np.interp(first['route_arc'],arc,headings),headings[mask],
                 np.interp(stop,arc,headings)]
    turning=float(np.sum(np.abs(np.diff(np.unwrap(angles)))))
    sweep=2*radius*np.sin(min(turning,np.pi)/2.)
    lower=max(reference['median'],minimum,2*radius,abs(speed)*response_s)+sweep
    upper=max(reference['p75'],lower+reference['p75']-reference['p25'])
    available=float(arc[-1])-first['route_arc']
    # Prefer a crossing before the terminal hull footprint. If prediction is
    # short, keep a distinct exact-safe crossing available rather than invent
    # one beyond the prediction or fall back to an almost coincident second.
    cap=max(minimum,available-radius)
    upper=min(upper,cap,available)
    lower=min(lower,upper)
    return dict(minimum=minimum,retention_minimum=footprint,lower=lower,upper=upper,
                available=available,first_end=first_end)


def second_gap_separated(first,second,band,retained=False):
    """Partial/X overlays are allowed; proximity is measured along the route."""
    # Acquisition clears the entire first local group. Once acquired, a new
    # presentation alternative extending that group must not evict a valid
    # second. Only actual first/second proximity is a retention release.
    minimum=band.get('retention_minimum',band['minimum']) if retained else band['minimum']
    return (gap_identity(first)!=gap_identity(second) and
            second['route_arc']-first['route_arc']>=minimum-1e-8)


def select_second_gap(groups,first,path,headings,hull,speed,response_s,margin,profile=None):
    """Scan all real future regions for non-crossing representatives, then band."""
    if first is None:
        return None
    band=second_gap_band(groups,first,path,headings,hull,speed,response_s,margin,profile)
    if band['available']<band['minimum']:
        return None
    choices=[]
    for group in groups:
        members=group.get('presentation_candidates',(group,))
        if any(gap_identity(g)==gap_identity(first) for g in members):
            continue
        eligible=[g for g in members if second_gap_separated(first,g,band)
                  and g['route_arc']<=first['route_arc']+band['available']+1e-8
                  and not segments_intersect(first,g)]
        if not eligible:
            continue
        # Pick the future crossing region before its representative segment.
        # Its earliest eligible crossing anchors route order independently of
        # the locally preferred segment's width or midpoint proximity.
        distance=min(g['route_arc'] for g in eligible)-first['route_arc']
        outside=max(band['lower']-distance,0.,distance-band['upper'])
        g=select_presentation_gap(eligible)
        choices.append(((outside,abs(distance-band['upper']),distance,
                         presentation_key(g),segments_conflict(first,g,margin)),g))
    return min(choices,key=lambda item:item[0])[1] if choices else None


def _geometry(path):
    delta=np.diff(path,axis=0)
    length=np.linalg.norm(delta,axis=1)
    return delta,length,np.r_[0.,np.cumsum(length)]


def goal_route_location(path,goal,radius):
    """First goal visit's closest approach, in route order (display only)."""
    if len(path)<2:
        return None
    delta,length,arc=_geometry(path)
    fraction=np.clip(np.sum((goal-path[:-1])*delta,axis=1)/np.maximum(length*length,1e-12),0.,1.)
    projected=path[:-1]+fraction[:,None]*delta
    d2=np.sum((projected-goal)**2,axis=1)
    near=np.flatnonzero(d2<=radius*radius)
    if not len(near):
        return None
    first=last=int(near[0])
    while last+1<len(d2) and d2[last+1]<=radius*radius:
        last+=1
    index=first+int(np.argmin(d2[first:last+1]))
    return dict(route_arc=float(arc[index]+fraction[index]*length[index]),
                route_segment=index,pos=projected[index])


def clipped_display_route(path,first,second,goal,goal_radius):
    """Clip at goal even without GAPs; original rollout is never written."""
    goal_event=goal_route_location(path,goal,goal_radius)
    waypoint=second if second is not None else first
    goal_wins=goal_event is not None and (waypoint is None or
        goal_event['route_arc']<=waypoint['route_arc']+1e-8)
    end=goal_event if goal_wins else waypoint
    if end is None:
        return path.copy(),None,first,second
    visible_first=first if first is not None and first['route_arc']<=end['route_arc']+1e-8 else None
    visible_second=second if second is not None and second['route_arc']<=end['route_arc']+1e-8 else None
    # Insert the actual segment projection BEFORE the exact center endpoint.
    # No later rollout knot survives; replacing only the last knot is wrong.
    prefix=np.vstack((path[:end['route_segment']+1],end['pos']))
    if goal_wins and np.linalg.norm(prefix[-1]-goal)>1e-8:
        prefix=np.vstack((prefix,goal))
    keep=np.r_[True,np.linalg.norm(np.diff(prefix,axis=0),axis=1)>1e-8]
    prefix=prefix[keep]
    before,after=split_display_route(prefix,visible_first,visible_second)
    return before,after,visible_first,visible_second
