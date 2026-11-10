"""First warning per tracked encounter, with walking/road context gates.

No file names, labels or annotated times are read by this policy. Ground ranges
are approximate projections under camera-height/focal assumptions, not depth
measurements. Every rejected candidate has a concrete diagnostic reason.
"""
from dataclasses import dataclass,asdict,replace
import math
import numpy as np
from bbox_evidence import Evidence
from object_tracks import iou,warp,warp_dense,same_object,same_vertical_object,separate_post_and_vehicle
from simple_trigger import clock_direction

@dataclass(frozen=True)
class HazardConfig:
    camera_height_m: float=1.7
    corridor_half_m: float=.95
    lookahead_s: float=3.
    min_lookahead_m: float=5.5
    max_lookahead_m: float=5.5
    min_span_s: float=.35
    min_observations: int=3
    min_confidence: float=.40
    growth_per_s: float=.10
    confirm_s: float=0.
    entry_speed: float=.18
    road_fraction: float=.60
    stationary_speed: float=.30
    memory_s: float=4.
    tall_obstacle_bottom: float=.64
    compact_obstacle_height: float=.12
    person_min_height: float=.18
    two_wheel_min_height: float=.12
    suppress_wearer: bool=True
    evidence_history_s: float=.85
    evidence_max_gap_s: float=.4
    evidence_stale_s: float=2.
    noaux_mode: bool=False
    path_min_overlap: float=.15
    feedback2: bool=False

def corridor_geometry(box,ctx,w,h,cfg):
    x1,y1,x2,y2=box;f=ctx['focal_px'];horizon=ctx['horizon']
    heading=ctx['path_heading_rad'] if ctx.get('path_valid') else 0.
    cx=w/2+f*math.tan(heading)
    # A straight trajectory through the camera's ground position has a shifted
    # image centre at ALL depths when the camera looks away from travel direction.
    bottom=max(y2,horizon+1.);z=f*cfg.camera_height_m/(bottom-horizon)
    look=float(np.clip(max(.6,ctx['walk_mps'])*cfg.lookahead_s,cfg.min_lookahead_m,cfg.max_lookahead_m))
    half=cfg.corridor_half_m*(bottom-horizon)/cfg.camera_height_m
    left,right=cx-half,cx+half
    overlap=max(0.,min(x2,right)-max(x1,left))/max(1.,min(x2-x1,2*half))
    in_path=overlap>=cfg.path_min_overlap and y2>horizon+.025*h
    top=max(horizon+1.,horizon+f*cfg.camera_height_m/look)
    top=min(top,.96*h);tophalf=cfg.corridor_half_m*(top-horizon)/cfg.camera_height_m
    nearhalf=cfg.corridor_half_m*(h-horizon)/cfg.camera_height_m
    poly=[[cx-nearhalf,h],[cx+nearhalf,h],[cx+tophalf,top],[cx-tophalf,top]]
    return dict(in_path=bool(in_path),overlap=overlap,ground_range_estimate_m=float(z),lookahead_m=look,
                corridor_center_x=cx,footprint=[left,y2,right,y2],polygon=poly,
                side=((x1+x2)/2-cx)/min(w,h))

class HazardPolicy:
    def __init__(self,config=None):
        self.config=config or HazardConfig()
        self.evidence=Evidence(history=self.config.evidence_history_s,
            max_gap_s=self.config.evidence_max_gap_s,stale_s=self.config.evidence_stale_s)
        self.states={};self.memory=[]
    def step(self,f,ctx,backflow=None):
        cfg=self.config;t=f['time_s'];w,h=f['width'],f['height'];events=[];debug=[]
        visible_ids={d.get('track_id') for d in f['detections']}
        visible_boxes={d.get('track_id'):d['box'] for d in f['detections']}
        def owner_visible(e):
            return any(i in visible_boxes and (iou(e['box'],visible_boxes[i])>.2 or same_object(e['box'],visible_boxes[i])) for i in e.get('aliases',[e['track_id']]))
        def compatible(a,b):
            # A large road-vehicle box must not consume an unrelated pole.
            # Two-wheel/obstacle switches remain allowed for partial detections.
            if a in (4,5,6) or b in (4,5,6):return a in (4,5,6) and b in (4,5,6)
            if cfg.feedback2 and a in (0,1,2,3) and b in (0,1,2,3):return True
            return (a==0)==(b==0)
        def identity_safe(r):
            x1,y1,x2,y2=r['box']
            if r.get('veto') or (x2-x1)*(y2-y1)>.5*w*h:return False
            if r.get('body_flat_fraction',0)>.85:return False
            if y1>.45*h and y2>.88*h:return False
            return True
        def same_encounter(e,r):
            if cfg.feedback2:
                component_link=(r['class_id'] in (1,2,3) and e['class_id'] in (1,2,3)
                    and any(same_object(e['box'],b) for b in r.get('component_boxes',[])))
                if r['track_id'] in e.get('distinct_ids',[]) and not component_link:return False
                if (e['class_id']==0)!=(r['class_id']==0):
                    return iou(e['box'],r['box'])>.55
                return same_object(e['box'],r['box']) or r['class_id']==7 and e.get('class_id')==7 and same_vertical_object(e['box'],r['box'])
            if cfg.noaux_mode and r.get('track_count',0)>3 and r.get('track_age',0)>.6:return False
            # An independently tracked object already visible before another
            # object's alert cannot inherit that alert merely by moving into
            # its old image position (e.g. consecutive parked vehicles).
            if cfg.noaux_mode and t-r.get('track_age',0.)<=e.get('warned_at',-math.inf)+1e-8:return False
            if e.get('identity_box') and r.get('identity_box') and same_object(e['identity_box'],r['identity_box']):return True
            return same_object(e['box'],r['box']) or r['class_id']==7 and e.get('class_id')==7 and same_vertical_object(e['box'],r['box'])
        for e in self.memory:
            for key in ('box','identity_box'):
                if e.get(key):e[key]=warp_dense(e[key],backflow,w,h) if backflow is not None and not ctx.get('camera_shock') else warp(e[key],f.get('matrix'))
        self.memory=[e for e in self.memory if t-e['last']<=cfg.memory_s]
        if cfg.feedback2:
            # Remember genuine co-visible neighbours before their boxes grow
            # into an old warning position. Mere later overlap is not identity.
            for e in self.memory:
                owners=[visible_boxes[i] for i in e.get('aliases',[e['track_id']]) if i in visible_boxes]
                for d in f['detections']:
                    if not owners or d['track_id'] in e.get('aliases',[]):continue
                    if all(not same_object(b,d['box']) and not (d['class_id']==e['class_id']==7 and same_vertical_object(b,d['box'])) for b in owners):
                        if d['track_id'] not in e['distinct_ids']:e['distinct_ids'].append(d['track_id'])
        for tid,s in list(self.states.items()):
            if t-s['last']>30:del self.states[tid]
        for r in self.evidence.step(f,ctx):
            tid=r['track_id'];cid=r['class_id'];geometry_box=r['box']
            raw_area_growth=r['growth']
            if cfg.noaux_mode and cid in (4,5,6) and (r['box'][0]<.01*w or r['box'][2]>.99*w):
                # Width grows when a sideways car emerges from the image edge.
                # Use height expansion to avoid calling this depth approach.
                r['growth']=min(r['growth'],2*r['height_growth'])
            if cid in (4,5,6) and r.get('vehicle_contact_span'):
                contact=r['vehicle_contact_span'];geometry_box=[contact[0],r['box'][1],contact[2],contact[1]]
            geometry_cfg=replace(cfg,corridor_half_m=.80,path_min_overlap=.35) if cfg.feedback2 and cid==7 else cfg
            geo=corridor_geometry(geometry_box,ctx,w,h,geometry_cfg)
            s=self.states.setdefault(tid,dict(last=t,fired=False,since=None));s['last']=t
            if r.get('wearer_equipment_candidate'):s['started_bottom_clipped']=True
            if r.get('bottom_obstacle_fragment'):s['started_bottom_obstacle']=True
            if not r.get('wearer_equipment_candidate') and (not cfg.noaux_mode or r['box'][3]<.95*h):s['seen_unclipped']=True
            # Transfer identity while the reappearing box still overlaps the
            # remembered object, even before it is near enough to warn. Waiting
            # for a second risk threshold lets a partial box drift too far away.
            identity_valid=identity_safe(r)
            if not s['fired'] and identity_valid:
                prior=next((e for e in self.memory
                    if (cfg.feedback2 or not owner_visible(e))
                    and not separate_post_and_vehicle(e,r,h)
                    and compatible(cid,e['class_id'])
                    and same_encounter(e,r)),None)
                if prior is not None:
                    prior.setdefault('aliases',[prior['track_id']]).append(tid);prior['last']=t;s['fired']=True
            reason=None;trigger=None
            distance=geo['ground_range_estimate_m'];near=distance<=geo['lookahead_m']
            large_near_unknown=(cid==7 and not ctx.get('path_valid') and r['box'][1]<.05*h and r['box'][3]>.45*h
                                and .12*w<r['box'][2]-r['box'][0]<.4*w and r.get('sidewalk_fraction',0)>.6
                                and r.get('body_vegetation_fraction',0)>.45)
            noaux_large_post=(cfg.noaux_mode and cid==7 and not ctx.get('path_valid')
                and ctx.get('walk_valid') and ctx['walk_mps']<cfg.stationary_speed
                and r['n']>=4 and r['span']>=.6-1e-8 and r.get('median_confidence',0)>=.55
                and r['box'][1]<.03*h and r['box'][3]>.4*h and r['aspect']>=2.5
                and .15*w<r['box'][2]-r['box'][0]<.35*w and r['speed']<.1)
            large_near_unknown=large_near_unknown or noaux_large_post
            # A slender post can have a truncated bottom. Require repeated current
            # detections and width expansion rather than a single weak score.
            temporal_post=(cfg.noaux_mode and cid==7 and r['n']>=4 and r['span']>=.6-1e-8
                and r.get('median_confidence',0)>=.24 and r['aspect']>=3.
                and r['box'][2]-r['box'][0]>=.025*w and r['box'][3]>.60*h and r['box'][1]<.10*h
                and r['width_growth']>.08 and r['speed']<.1 and distance<=7.5)
            # A narrow, fully visible post beside the projected centre can
            # deserve a proximity caution as it expands consistently. This
            # does not turn road traffic or broad boundaries into hazards.
            side_post=(cfg.noaux_mode and cid==7 and r['n']>=4 and r['span']>=.6-1e-8
                and r.get('median_confidence',0)>=.4 and r['aspect']>=2.5
                and .025*w<=r['box'][2]-r['box'][0]<=.10*w
                and r['box'][1]>.1*h and r['box'][3]>.60*h
                and r['box'][3]-r['box'][1]>.12*h
                and r['width_growth']>.12 and r['height_growth']>.08 and r['speed']<.1
                and distance<=7.5 and abs(geo['side'])*min(w,h)*distance/ctx['focal_px']<=1.6)
            if cfg.feedback2:
                # A path-side caution still needs actual corridor overlap.
                side_post=side_post and geo['overlap']>=.5
                temporal_post=temporal_post and r['bw']>=.08
            if side_post:near=True
            if temporal_post:near=True
            if large_near_unknown:near=True
            # Large road vehicles have an independent apparent-size cue. A
            # parked car may obstruct the path even with slow area expansion.
            if cid in (4,5,6) and r['box'][2]-r['box'][0]>=.25*w and r['aspect']<.5 and r['bh']>=.20 and r['growth']>=.15 and distance<=12.:near=True
            if cid in (1,2,3) and r['bh']>=.14 and r['bw']>=.18 and r['aspect']<1. and r['confidence']>=.5 and r['growth']>=.15 and distance<=8. and (not cfg.feedback2 or r['box'][3]-r['box'][1]>=.17*h):near=True
            if cid==7 and r.get('public_class') in (13,56,58) and r['bh']>=.2 and r['box'][2]-r['box'][0]>.18*w and r['growth']>=.15 and distance<=7.:near=True
            if cid==0 and r['bh']>=.4 and r['growth']>=.4 and distance<=8.:
                near=True
                if geo['overlap']>.05:geo['in_path']=True
            if r.get('semantic_kind')=='horizontal_barrier' and r['box'][2]-r['box'][0]>.5*w and r['box'][3]>.3*h:near=True
            fast_frontal=False
            if cfg.feedback2:
                # Looming is independent of the noisy ground/horizon projection.
                # Both dimensions must expand, with little bearing change. This
                # does not treat lateral emergence from an edge as collision TTC.
                x1,y1,x2,y2=r['box'];hg=r['height_growth'];wg=r['width_growth']
                fast_frontal=(cid in (4,5,6) and ctx.get('path_valid') and abs(ctx['path_heading_rad'])<math.radians(15)
                    and r['n']>=4 and r['span']>=.6-1e-8
                    and r.get('median_confidence',0)>=.60 and .5<=r['aspect']<=1.8
                    and x1>.02*w and x2<.98*w and (y2-y1)>=.075*h and (x2-x1)>=.14*w
                    and .25*w<=(x1+x2)/2<=.80*w and hg>=.33 and wg>=.25
                    and abs((x1+x2)/2-geo['corridor_center_x'])/ctx['focal_px']<=.16
                    and .5<=wg/max(hg,1e-6)<=2. and abs(r['vx'])<.35*min(hg,wg))
                if cid in (4,5,6):
                    # Broad near vehicles may lack reliable ground projection;
                    # require appreciable vertical extent, not bus width alone.
                    apparent_near=(r['box'][2]-r['box'][0]>=.25*w and (r['box'][3]-r['box'][1])>=.20*h
                        and r['aspect']<.5 and r['growth']>=.15 and distance<=12.)
                    near=distance<=geo['lookahead_m'] or apparent_near or fast_frontal
                if cid in (1,2,3) and distance<=5.5:near=True
            observed=r['n']>=cfg.min_observations and r['span']>=cfg.min_span_s and r['observed']
            confidence=r.get('recent_peak_confidence',r.get('peak_confidence',r['confidence']))>=cfg.min_confidence or temporal_post
            if r.get('source','').startswith('grounding'):confidence=r['confidence']>=.40 or r['confidence']>=.25 and r.get('grounding_confirmations',0)>=2
            stationary=ctx.get('stationary') or ctx.get('walk_valid') and ctx['walk_mps']<cfg.stationary_speed
            road_fraction=r.get('vehicle_road_fraction',r.get('road_fraction',0)) if cid in (4,5,6) else r.get('road_fraction',0)
            sidewalk_fraction=r.get('vehicle_sidewalk_fraction',r.get('sidewalk_fraction',0)) if cid in (4,5,6) else r.get('sidewalk_fraction',0)
            if cid in (4,5,6):
                # A detector box can extend over the sidewalk while the actual
                # vehicle's wheels are entirely on the road. Strong segmented
                # contact evidence takes precedence over the loose rectangle.
                if not (r.get('vehicle_road_fraction',0)>=.8 and r.get('vehicle_sidewalk_fraction',1)<=.1):
                    road_fraction=max(road_fraction,r.get('road_fraction',0));sidewalk_fraction=max(sidewalk_fraction,r.get('sidewalk_fraction',0))
            road=road_fraction>=cfg.road_fraction and road_fraction>2*sidewalk_fraction
            on_road=ctx.get('user_surface')==2 and ctx.get('surface_confidence',0)>.55 and ctx.get('path_valid',False)
            inward=-np.sign(geo['side'])*r['vx']
            if cfg.feedback2 and cid in (1,2,3) and observed and confidence and r['growth']>=cfg.growth_per_s and r['height_growth']>.05:
                s['last_approach_evidence']=t
                s['approach_area']=(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1])
            recent_two_wheel_approach=(cfg.feedback2 and cid in (1,2,3)
                and t-s.get('last_approach_evidence',-math.inf)<=.6+1e-8
                and (r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1])>=.8*s.get('approach_area',math.inf))
            if r.get('veto'):reason=r['veto']
            elif cid==0 and (r['box'][0]<=.005*w or r['box'][2]>=.995*w) and (r['box'][2]-r['box'][0])<.09*w:reason='peripheral_person_fragment'
            elif cfg.suppress_wearer and cid==0 and r['box'][3]>=.88*h and r['box'][1]>.45*h:reason='wearer_body_fragment'
            elif cfg.suppress_wearer and cid in (1,2,3,4,5,6) and r['box'][3]>=.9*h and r['box'][1]>.5*h and r.get('body_person_fraction',0)>.15:reason='wearer_body_fragment'
            elif cid==7 and r.get('source')=='surface' and r['box'][1]>.6*h and r['box'][3]-r['box'][1]<.2*h and r['box'][2]-r['box'][0]<.12*w and r.get('body_flat_fraction',0)>.5:reason='small_semantic_fragment'
            elif cid==7 and r.get('source')=='public' and r.get('public_class') in (13,56,58) and r['box'][1]>.6*h and r.get('foot_ground_support',1)<.1 and r.get('body_person_fraction',0)>.2:reason='partial_object_without_ground_contact'
            elif cfg.suppress_wearer and cid==7 and r['box'][1]>.5*h and r['box'][3]>.8*h and (r.get('body_person_fraction',0)>.1 or r.get('body_flat_fraction',0)>.7) and r.get('body_solid_fraction',0)<.05:reason='wearer_equipment'
            elif cid==7 and r.get('body_flat_fraction',0)>.85 and r.get('body_solid_fraction',0)<.05 and not (r.get('semantic_kind')=='horizontal_barrier' and r['confidence']>=.45):reason='flat_ground_detection'
            elif s['fired']:reason='already_warned'
            elif not confidence:reason='low_confidence'
            elif not observed:reason='insufficient_history'
            elif cfg.noaux_mode and cid in ((1,2,3,4,5,6) if cfg.feedback2 else (4,5,6)) and stationary and distance>3. and abs(r['vx'])>.06 and (r['growth']<.35 or abs(r['vx'])>max(.08,r['growth']*.25) or abs(r['vx'])>.35 and abs(r['vx'])>2*abs(r['vy'])):reason='stationary_lateral_traffic'
            elif cfg.noaux_mode and cid in ((1,2,3,4,5,6) if cfg.feedback2 else (1,2,3)) and s.get('started_bottom_clipped') and not s.get('seen_unclipped'):reason='bottom_clipped_two_wheel_unconfirmed'
            elif cfg.feedback2 and cid==7 and s.get('started_bottom_obstacle') and not s.get('seen_unclipped'):reason='unconfirmed_bottom_obstacle_fragment'
            elif cfg.feedback2 and cid in (4,5,6) and (r['box'][0]<.005*w or r['box'][2]>.995*w) and ((r['box'][0]+r['box'][2])/2<.18*w or (r['box'][0]+r['box'][2])/2>.82*w) and inward<cfg.entry_speed:reason='peripheral_passing_vehicle'
            elif cfg.feedback2 and cid==7 and (r['box'][2]<.22*w or r['box'][0]>.78*w):reason='peripheral_obstacle'
            elif cid==7 and r.get('body_boundary_fraction',0)>.45 and r['aspect']<3. and r['box'][2]-r['box'][0]>.14*w:reason='continuous_boundary'
            elif cid==7 and r.get('body_vegetation_fraction',0)>.75 and r['aspect']<2.5 and r['box'][2]-r['box'][0]>.2*w:reason='continuous_vegetation'
            elif cid==7 and r.get('source')=='surface' and r.get('semantic_kind')=='vegetation' and r.get('body_flat_fraction',0)>.6 and r.get('body_vegetation_fraction',0)<.4:reason='sparse_vegetation_on_ground'
            elif cfg.noaux_mode and cid==7 and (r['box'][0]<.005*w or r['box'][2]>.995*w) and (geo['overlap']<.5 or (r['box'][0]+r['box'][2])/2<(.18 if cfg.feedback2 else .12)*w or (r['box'][0]+r['box'][2])/2>(.82 if cfg.feedback2 else .88)*w):reason='peripheral_obstacle_fragment'
            elif not geo['in_path'] and not large_near_unknown and not side_post and not fast_frontal:reason='outside_walking_path'
            elif cid in (7,8) and road and ctx.get('user_surface')==1:reason='roadside_obstacle'
            elif cid in (4,5,6) and road and (stationary or not on_road) and not (distance<2.5 and r['growth']>.4):reason='road_traffic'
            elif cid==0 and r.get('body_person_fraction',0)<.08 and r.get('body_flat_fraction',0)>.65:reason='unconfirmed_person_on_ground'
            elif cid==0 and r.get('surface_known') and r.get('body_person_fraction',0)<.05 and r.get('body_vehicle_fraction',0)>.65:reason='vehicle_misclassified_as_person'
            elif cid==0 and (r['box'][3]-r['box'][1])/h<cfg.person_min_height:reason='person_too_small_for_near_warning'
            elif cid in (1,2,3) and (r['box'][3]-r['box'][1])/h<cfg.two_wheel_min_height:reason='two_wheel_too_small_for_near_warning'
            elif cid in (1,2,3) and r.get('body_flat_fraction',0)>.75 and r.get('body_vehicle_fraction',0)<.05:reason='unconfirmed_two_wheel_on_ground'
            elif cid==7 and r.get('semantic_kind')!='horizontal_barrier' and (r['box'][3]-r['box'][1])/h<cfg.compact_obstacle_height:reason='obstacle_too_small_for_near_warning'
            elif cid==7 and r['aspect']>=2.5 and (r['box'][3]-r['box'][1])>.35*h and r['box'][3]<cfg.tall_obstacle_bottom*h and not large_near_unknown and not temporal_post:reason='tall_obstacle_not_near'
            elif cid in (7,8) and r['box'][3]<.55*h and not large_near_unknown and r.get('semantic_kind')!='horizontal_barrier':reason='insufficient_near_field_support'
            elif not near:reason='too_far'
            elif cid in (7,8) and r['speed']>.35:reason='unstable_object_geometry'
            else:
                # Lateral entry and approach share all semantic/path/range gates.
                if fast_frontal:trigger='fast_frontal_approach'
                elif large_near_unknown:trigger='large_near_obstruction_unknown_path'
                elif side_post:trigger='near_side_obstacle'
                elif inward>=cfg.entry_speed and r['bh']>=.20 and r['camera_ok']:trigger='sudden_entry'
                elif r['growth']>=cfg.growth_per_s:trigger='approach'
                elif recent_two_wheel_approach:trigger='approach_recent_expansion'
                elif distance<3.5 and ctx['walk_mps']>.4 and r['growth']>-.12:trigger='near_obstruction'
                else:reason='no_approach_evidence'
            if trigger:
                if s['since'] is None:s['since']=t
                if t-s['since']<cfg.confirm_s-1e-8:reason='confirming'
                else:
                    # Two separately visible tracks must not steal each other's
                    # warning. Reassociation applies only after the old track
                    # disappeared; detector fusion already removes duplicates
                    # within a frame. Keep all aliases alive after an ID split.
                    duplicate=next((e for e in self.memory
                        if (cfg.feedback2 or not owner_visible(e))
                        and not separate_post_and_vehicle(e,r,h)
                        and compatible(cid,e['class_id'])
                        and same_encounter(e,r)),None)
                    if duplicate:
                        reason='already_warned_reassociated';s['fired']=True;duplicate['last']=t
                        duplicate.setdefault('aliases',[duplicate['track_id']]).append(tid)
                    else:
                        s['fired']=True;reason='warned'
                        keys=['box','class_id','class','confidence','track_id','frame','time_s']
                        e={k:r[k] for k in keys};e.update(event='approach',rule_group=trigger,clock=clock_direction(r['box'],w,ctx['focal_px']),
                            source=r.get('source','custom'),growth_per_second=r['growth'],walk_mps=ctx['walk_mps'],
                            path_valid=ctx['path_valid'],path_heading_deg=math.degrees(ctx['path_heading_rad']),
                            roi_polygon=geo['polygon'],roi_overlap=geo['overlap'],
                            ground_range_estimate_m=None if noaux_large_post else distance,range_is_metric_measurement=False,
                            near_evidence='apparent_size_unknown_path' if noaux_large_post else 'near_side_expansion' if side_post else 'repeated_post_expansion' if temporal_post else 'projection',
                            priority='urgent' if trigger=='sudden_entry' else 'normal')
                        distinct=[d['track_id'] for d in f['detections'] if d['track_id']!=tid and not same_object(d['box'],r['box']) and not (d['class_id']==cid==7 and same_vertical_object(d['box'],r['box']))]
                        events.append(e);self.memory.append(dict(box=r['box'],identity_box=r.get('identity_box'),last=t,warned_at=t,track_id=tid,class_id=cid,semantic_kind=r.get('semantic_kind'),aliases=[tid],distinct_ids=distinct))
            else:s['since']=None
            if s['fired'] and identity_valid:
                for e in self.memory:
                    if tid in e.get('aliases',[e['track_id']]):
                        if r.get('identity_box'):e['identity_box']=r['identity_box']
                        old=np.asarray(e['box']);new=np.asarray(r['box'])
                        oldarea=np.maximum(0,old[2:]-old[:2]).prod();newarea=np.maximum(0,new[2:]-new[:2]).prod()
                        if newarea<.5*oldarea and same_object(old,new):
                            e.update(box=np.r_[np.minimum(old[:2],new[:2]),np.maximum(old[2:],new[2:])].tolist(),last=t)
                        elif iou(old,new)>.2 or same_vertical_object(old,new):e.update(box=r['box'],last=t)
                        # A tracker can jump to a different distant instance.
                        # Do not teleport the warning memory with that ID; its
                        # box continues to follow local image motion instead.
            debug.append(dict(track_id=tid,**{'class':r['class']},class_id=cid,box=r['box'],reason=reason,
                              observations=r['n'],history_span_s=r['span'],observation_gap_s=r['gap'],history_ready=bool(observed),
                              fast_frontal=bool(fast_frontal),noaux_large_post=bool(noaux_large_post),temporal_post=bool(temporal_post),side_post=bool(side_post),median_confidence=r.get('median_confidence'),width_growth=r['width_growth'],growth=r['growth'],raw_area_growth=raw_area_growth,height_growth=r['height_growth'],inward_speed=float(inward),raw_speed=r['raw_speed'],residual_speed=r['speed'],vx=r['vx'],vy=r['vy'],confidence=r['confidence'],source=r.get('source','custom'),
                              road_fraction=road_fraction,sidewalk_fraction=sidewalk_fraction,**geo))
        return events,debug
