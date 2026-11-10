"""Team PT only: image geometry, current detections and causal tracking.

No surface model, public detector, text detector, or semantic cache is loaded.
Image-space path membership is an approximation, not road classification.
"""
from dataclasses import replace
import copy, time
import cv2, numpy as np
from object_tracks import ObjectTracker, iou, same_vertical_object, same_object
from hazard_policy import HazardConfig,HazardPolicy
from scene_context import SceneContext,SurfaceFlow
from image_motion import ImageMotion
from walking_direction import WalkingDirection
from ego_motion import EgoMotion
from mobility_state import MobilityConfig,MobilityEstimator,annotate_mobility
from risk_message import describe_risk
from vendor.light_online import LightRule
from navigation_config import LIGHT_CONFIG
from vision_features import color_features,image_quality

def noaux_config(camera_mode='egocentric',**kwargs):
    defaults=dict(feedback2=True,min_lookahead_m=5.0,max_lookahead_m=5.0,compact_obstacle_height=.18)
    defaults.update(kwargs);kwargs=defaults
    return HazardConfig(suppress_wearer=camera_mode=='egocentric',noaux_mode=True,**kwargs)

def team_candidates(f,suppress_wearer=True):
    if not f.get('image_quality',{}).get('valid',True):return []
    w,h=f['width'],f['height'];out=[]
    # Only raw detections from this model are accepted. No transported boxes
    # become additional evidence observations.
    for raw in sorted(f['detections'],key=lambda d:d['confidence'],reverse=True):
        allowed=('box','confidence','class_id','class','source','color_features','vehicle_contact_span')
        d={k:copy.deepcopy(raw[k]) for k in allowed if k in raw};cid=d['class_id'];b=d['box'];x1,y1,x2,y2=b
        if d.get('source','custom') not in ('custom','custom_crop'):raise ValueError('Auxiliary detections are forbidden')
        if d['confidence']<(.08 if cid==9 else .18):continue
        d.update(source=d.get('source','custom'),surface_known=False)
        if suppress_wearer:
            if cid==0 and ((y2>.92*h and x2-x1>.4*w and (y2-y1)/max(1,x2-x1)<1.2) or
                          (y2>=.90*h and y1>.52*h and (x1<.03*w or x2>.97*w))):d['veto']='wearer_body_fragment'
            if cid in (1,2,3,4,5,6) and y2>.97*h and y1>.50*h:
                d['wearer_equipment_candidate']=True
            if cid==7 and y2>.97*h and y1>.45*h:
                d['bottom_obstacle_fragment']=True
        # A high-scoring upper pole fragment must not erase the model's
        # lower-scoring full-height detection. Merge measured extents only;
        # this is still one observation from the current source frame.
        vertical=next((x for x in out if cid==x['class_id']==7 and same_vertical_object(b,x['box'])),None)
        if vertical is not None:
            vertical['box']=[min(b[0],vertical['box'][0]),min(b[1],vertical['box'][1]),
                             max(b[2],vertical['box'][2]),max(b[3],vertical['box'][3])]
            continue
        light_duplicate=next((x for x in out if cid==x['class_id']==9 and same_object(b,x['box'])),None)
        if light_duplicate is not None:
            def supported_shape(z):
                a=z['box'];return (a[2]-a[0])/max(1,a[3]-a[1])<=LIGHT_CONFIG['maxaspect']
            if supported_shape(d) and not supported_shape(light_duplicate):
                out[out.index(light_duplicate)]=d
            continue
        if any(cid==x['class_id'] and iou(b,x['box'])>.55 for x in out):continue
        # Family-level duplicate boxes, never absorb a separate pole/person.
        if cid in (1,2,3) and any(x['class_id'] in (1,2,3) and iou(b,x['box'])>.65 for x in out):continue
        out.append(d)
    # Whole/part boxes from the same model can have different class IDs and
    # different confidences. Keep a single measured extent, not extra evidence.
    merged=[]
    for d in sorted(out,key=lambda x:(x['box'][2]-x['box'][0])*(x['box'][3]-x['box'][1]),reverse=True):
        b=np.asarray(d['box']);sz=b[2:]-b[:2];discard=False
        for x in merged:
            a=np.asarray(x['box']);sa=a[2:]-a[:2]
            inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))
            cover=inter.prod()/max(1.,min(sa.prod(),sz.prod()))
            family=d['class_id'] in (1,2,3) and x['class_id'] in (1,2,3)
            same_person=d['class_id']==x['class_id']==0
            whole_part=(family or same_person) and cover>.85 and abs(a[3]-b[3])<.25*max(sa[1],sz[1])
            # Front wheel and frame can be returned as adjacent partial bicycle
            # boxes. Require shared vertical extent, an overlap, and a narrow part.
            bicycle_parts=(d['class_id']==x['class_id']==1 and inter[0]>.15*min(sa[0],sz[0])
                and inter[1]>.75*min(sa[1],sz[1]) and min(sa[0],sz[0])<.65*max(sa[0],sz[0])
                and abs(a[3]-b[3])<.25*max(sa[1],sz[1]))
            if whole_part or bicycle_parts:
                x.setdefault('component_boxes',[x['box']]);x['component_boxes'].append(d['box'])
                x['box']=np.r_[np.minimum(a[:2],b[:2]),np.maximum(a[2:],b[2:])].tolist()
                x['confidence']=max(x['confidence'],d['confidence']);discard=True;break
        if not discard:merged.append(d)
    # A full upright person hypothesis takes precedence over an almost identical
    # two-wheel box. A bicycle extending below/around a rider is not removed.
    for d in merged:
        if d['class_id'] not in (1,2,3):continue
        b=np.asarray(d['box']);sz=b[2:]-b[:2]
        for p in merged:
            if p['class_id']!=0 or p['confidence']<.4:continue
            a=np.asarray(p['box']);sa=a[2:]-a[:2]
            if sa[1]/max(sa[0],1)<1.3:continue
            cover=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()/max(1.,sz.prod())
            if cover>.8 and sa[1]>.8*sz[1] and sa[0]>.6*sz[0] and abs(a[3]-b[3])<.25*max(sa[1],sz[1]):
                d['veto']='overlapping_person_hypothesis';break
    return merged

class SingleModelCore:
    def __init__(self,config=None,light_config=None):
        self.policy=HazardPolicy(config or noaux_config())
        self.lights=LightRule(**(light_config or LIGHT_CONFIG))
        self.scene=SceneContext();self.tracker=ObjectTracker()
        self.mobility=MobilityEstimator(MobilityConfig(
            window_s=max(1.4,self.policy.config.evidence_history_s),
            max_gap_s=max(.45,self.policy.config.evidence_max_gap_s)))
    def step_observations(self,f,dense=None):
        # Build context anew. Fused historical context is deliberately ignored.
        ctx=self.scene.step(f,None,None)
        ctx.update(surface_available=False,geometry_mode='projection',noaux_mode=True)
        f=dict(f);f['detections']=self.tracker.step(f,team_candidates(f,self.policy.config.suppress_wearer),dense)
        events,decisions=self.policy.step(f,ctx,dense)
        events+=self.lights.step(f);self.lights.prune(f['time_s'])
        for e in events:
            if e['event'].startswith('light_'):
                e['message']='신호가 초록불로 바뀌었습니다.' if e['color']=='green' else '신호가 빨간불로 바뀌었습니다.'
            else:
                label={0:'사람',1:'자전거',2:'킥보드',3:'오토바이',7:'장애물',8:'장애물'}.get(e['class_id'],'차량')
                e['message']=f"{e['clock']}시 방향, {label}에 유의하세요."
        debug=dict(f,context=ctx,decisions=decisions,light_decisions=self.lights.diagnostics)
        annotate_mobility(self.mobility,debug,events)
        for e in events:describe_risk(e)
        return events,debug

class CropScheduler:
    """One crop per frame maximum; 0.4s search plus two real confirmation frames."""
    def __init__(self,interval=.4):
        self.interval=interval;self.last=-1e9;self.recent=[];self.index=0
        self.pending=None;self.cooldowns=[];self.scheduled=0;self.confirming=False
    def choose(self,ds,w,h,t):
        if t<=self.last:return None
        if self.pending is not None:
            deadline,remaining,roi=self.pending
            if remaining>0 and t<=deadline and t>self.last:
                self.pending=(deadline,remaining-1,roi);self.last=t;self.confirming=True
                return dict(roi,kind='confirm_current_detection')
            self.pending=None
        if t-self.last<self.interval-1e-8:return None
        self.last=t;self.recent=[(ts,b) for ts,b in self.recent if t-ts<1.2];self.confirming=False
        self.scheduled+=1
        candidates=[]
        for d in ds:
            b=d['box'];bw=b[2]-b[0];bh=b[3]-b[1]
            if d['class_id'] not in (1,2,3,7,9) or max(bw,bh)>.50*max(w,h):continue
            if d['class_id']!=9 and d['confidence']>.55:continue
            if any(iou(b,old)>.25 for _,old in self.recent):continue
            score=(2. if d['class_id']==9 else 1.)+d['confidence']
            candidates.append((score,d))
        # Ensure candidates cannot indefinitely starve the unobserved path.
        if candidates and self.scheduled%3!=0:
            d=max(candidates,key=lambda x:x[0])[1];b=d['box'];cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
            side=min(max(2.*max(b[2]-b[0],b[3]-b[1]),.5*min(w,h)),.65*max(w,h))
            self.recent.append((t,b));kind='candidate'
        else:
            # General path search; no video-specific time or object hints.
            cx=w*(.38 if self.index%2==0 else .62);cy=h*(.52 if self.index%4<2 else .72)
            side=.70*min(w,h);self.index+=1;kind='path_search'
        cw=min(w,side);ch=min(h,side)
        x=max(0,min(w-cw,cx-cw/2));y=max(0,min(h-ch,cy-ch/2))
        return dict(box=[int(x),int(y),int(x+cw),int(y+ch)],kind=kind)
    def observe(self,extra,full,roi,t):
        if self.confirming:return
        self.cooldowns=[(ts,b) for ts,b in self.cooldowns if t-ts<1.2]
        for d in sorted(extra,key=lambda x:x['confidence'],reverse=True):
            if d['class_id'] not in (1,2,3,7,9) or d['confidence']<(.15 if d['class_id']==9 else .25):continue
            if any(a['class_id']==d['class_id'] and a['confidence']>=.25 and iou(a['box'],d['box'])>.4 for a in full):continue
            if any(iou(old,d['box'])>.25 for _,old in self.cooldowns):continue
            self.pending=(t+3.6,2,dict(roi));self.cooldowns.append((t,d['box']));break

class Pipeline:
    def __init__(self,weights,device='cpu',threads=2,camera_mode='egocentric',imgsz=960,
                 crop=True,crop_interval=.4,profile_stages=True,config=None,light_config=None):
        import torch
        from ultralytics import YOLO
        if camera_mode not in ('fixed','egocentric'):raise ValueError('Invalid camera_mode')
        self.torch=torch;self.device=device;self.threads=threads;self.camera_mode=camera_mode
        self.imgsz=imgsz;self.crop_enabled=crop;self.crop_interval=crop_interval
        self.profile_stages=profile_stages;self.custom=YOLO(weights)
        self.config=config or noaux_config(camera_mode);self.light_config=dict(light_config or LIGHT_CONFIG)
        self.grounding=None;self.public=None;self.surface=None;self.public_every=0
        torch.set_num_threads(threads);cv2.setNumThreads(1);self.reset()
    def reset(self):
        self.core=SingleModelCore(self.config,self.light_config);self.policy=self.core.policy;self.lights=self.core.lights
        self.motion=ImageMotion();self.direction=WalkingDirection();self.ego=EgoMotion();self.surfaceflow=SurfaceFlow()
        self.cropper=CropScheduler(self.crop_interval);self.count=0;self.last_time=None;self.last_shape=None
    def predict(self,img,size,offset=(0,0),source='custom'):
        self.torch.set_num_threads(self.threads)
        result=self.custom.predict(img,imgsz=size,conf=.08,iou=.65,device=self.device,verbose=False,rect=True)[0]
        self.torch.set_num_threads(self.threads)
        ds=[];ox,oy=offset
        polygons=result.masks.xy if result.masks is not None else []
        for i,b in enumerate(result.boxes.data.cpu().tolist()):
            cid=int(b[5]);box=[b[0]+ox,b[1]+oy,b[2]+ox,b[3]+oy]
            d=dict(box=box,confidence=b[4],class_id=cid,**{'class':self.custom.names[cid]},source=source)
            if i<len(polygons) and len(polygons[i])>=3 and cid in (4,5,6):
                poly=polygons[i]+np.array([ox,oy]);bottom=float(np.percentile(poly[:,1],98))
                foot=poly[poly[:,1]>=bottom-max(3,.04*(box[3]-box[1]))]
                if len(foot)>=2 and np.ptp(foot[:,0])>3:
                    d['vehicle_contact_span']=[float(np.min(foot[:,0])),bottom,float(np.max(foot[:,0])),bottom]
            ds.append(d)
        return ds
    def step(self,img,frame,time_s,surface_due=False,grounding_due=False):
        if not np.isfinite(time_s) or time_s<0:raise ValueError('Timestamp must be finite and nonnegative')
        if self.last_time is not None and time_s<=self.last_time:raise ValueError('Timestamps must increase')
        h,w=img.shape[:2]
        if self.last_shape is not None and (self.last_shape!=(h,w) or time_s-self.last_time>3.):self.reset()
        self.last_time=time_s;self.last_shape=(h,w)
        start=time.perf_counter();stamp=start;stages={}
        def mark(name):
            nonlocal stamp
            if self.device!='cpu':self.torch.cuda.synchronize('cuda:'+str(self.device))
            now=time.perf_counter();stages[name]=(now-stamp)*1000;stamp=now
        ds=self.predict(img,self.imgsz);mark('custom_yolo')
        roi=self.cropper.choose(ds,w,h,time_s) if self.crop_enabled else None
        if roi:
            x1,y1,x2,y2=roi['box'];extra=self.predict(img[y1:y2,x1:x2],640,(x1,y1),'custom_crop')
            # Reject truncated crop-border objects unless the full frame agrees.
            extra=[d for d in extra if (d['box'][0]>x1+3 and d['box'][1]>y1+3 and d['box'][2]<x2-3 and d['box'][3]<y2-3)
                   or any(d['class_id']==a['class_id'] and iou(d['box'],a['box'])>.4 for a in ds)]
            self.cropper.observe(extra,ds,roi,time_s)
            ds+=extra
        for d in ds:
            if d['class_id']==9:d['color_features']=color_features(img,d['box'])
        mark('custom_crop_color')
        boxes=[d['box'] for d in ds if d['class_id']<7 and d['confidence']>=.5]
        f=dict(frame=frame,time_s=time_s,width=w,height=h,detections=ds,image_quality=image_quality(img))
        if self.camera_mode=='fixed':
            f.update(matrix=[[1.,0.,0.],[0.,1.,0.]],camera={'camera_ok':True,'method':'fixed'})
            f['ego']=dict(focal_px=26/36*max(w,h),horizon=h/2,camera_cx=w/2,walk_mps=0.,walk_valid=True,camera_shock=False)
            f['direction']=dict(valid=False,stationary=True)
        else:
            matrix,camera,pairs=self.motion.step(img,boxes)
            f.update(matrix=None if matrix is None else matrix.tolist(),camera=camera)
            f['ego']=self.ego.step(f,pairs);f['direction']=self.direction.step(img,time_s,boxes)
        dense=self.surfaceflow.step(img) # Optical flow only; no segmentation network.
        if dense is not None:dense=dense.astype(np.float16).astype(np.float32)
        mark('motion')
        observations=copy.deepcopy(f)
        events,debug=self.core.step_observations(f,dense);mark('tracking_rules')
        elapsed=time.perf_counter()-start
        for e in events:e['processing_s']=elapsed
        debug.update(stage_ms=stages,processing_s=elapsed,crop=roi,model_count=1,
                     model_input_sizes={'custom':self.imgsz,'crop':640},raw_observations=observations)
        self.count+=1
        return events,debug
