"""Causal detector fusion and class-tolerant object association."""
import copy
import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

NAMES={0:'person',1:'bicycle',2:'scooter',3:'motorcycle',4:'car',5:'bus',6:'other_vehicle',7:'obstacle',8:'stairs',9:'traffic_light'}
COCO={0:0,1:1,2:4,3:3,5:5,7:6,9:9,13:7,56:7,58:7}

def iou(a,b):
    a=np.asarray(a);b=np.asarray(b);inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(1.,np.prod(a[2:]-a[:2])+np.prod(b[2:]-b[:2])-inter))

def same_object(a,b):
    a=np.asarray(a);b=np.asarray(b)
    if iou(a,b)>.4:return True
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    coverage=inter/max(1.,min(np.prod(a[2:]-a[:2]),np.prod(b[2:]-b[:2])))
    return bool(coverage>.72 and abs(a[3]-b[3])<.35*max(a[3]-a[1],b[3]-b[1]))

def same_vertical_object(a,b):
    a=np.asarray(a);b=np.asarray(b);sa=a[2:]-a[:2];sb=b[2:]-b[:2]
    if min(sa)<=0 or min(sb)<=0 or min(sa[1]/sa[0],sb[1]/sb[0])<2.5:return False
    overlap=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2]))/np.minimum(sa,sb)
    return bool(overlap[0]>.75 and overlap[1]>.6 and max(sa[0],sb[0])/min(sa[0],sb[0])<3.)

def separate_post_and_vehicle(a,b,h):
    """A railing post can occlude a vehicle with a different ground contact."""
    post,other=(a,b) if a.get('semantic_kind')=='guardrail_terminal_post' else (b,a)
    return (post.get('semantic_kind')=='guardrail_terminal_post'
            and other['class_id'] in (1,2,3,4,5,6)
            and abs(post['box'][3]-other['box'][3])>.025*h)

def warp(box,matrix):
    if matrix is None:return list(box)
    x1,y1,x2,y2=box;p=np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]]);m=np.asarray(matrix)
    q=p@m[:,:2].T+m[:,2];return np.r_[q.min(0),q.max(0)].tolist()

def warp_dense(box,flow,w,h):
    if flow is None:return list(box)
    sh,sw=flow.shape[:2];scale=np.array([sw/w,sh/h])
    x1,y1,x2,y2=box
    # Interior points follow the object; box corners often lie on background.
    points=np.array([[x1+(x2-x1)*a,y1+(y2-y1)*b] for a in (.2,.5,.8) for b in (.2,.5,.8)],np.float32)*scale
    q=points.copy()
    for _ in range(3):
        delta=cv2.remap(flow,q[:,0].astype(np.float32).reshape(-1,1),q[:,1].astype(np.float32).reshape(-1,1),cv2.INTER_LINEAR,borderMode=cv2.BORDER_REPLICATE).reshape(-1,2)
        q=points-delta
    delta=np.median(q-points,axis=0)/scale
    # Robust local expansion, without allowing a single bad flow to explode boxes.
    ratios=[]
    for axis in (0,1):
        a=points[:,axis]-np.median(points[:,axis]);b=q[:,axis]-np.median(q[:,axis])
        nz=np.abs(a)>1
        ratios.append(float(np.clip(np.median(b[nz]/a[nz]) if nz.any() else 1.,.85,1.20)))
    center=np.array([(x1+x2)/2,(y1+y2)/2])+delta
    half=np.array([(x2-x1)/2,(y2-y1)/2])*ratios
    return np.r_[center-half,center+half].tolist()

class DetectorFusion:
    def __init__(self,suppress_wearer=True):
        self.suppress_wearer=suppress_wearer
        self.public=[];self.stamp=-1e9;self.grounding=[];self.ground_stamp=-1e9;self.scooters=[];self.scooter_stamp=-1e9
    def step(self,f,scene,backflow=None):
        t=f['time_s'];w,h=f['width'],f['height']
        if not f.get('image_quality',{}).get('valid',True):
            self.public=[];self.grounding=[];self.scooters=[];self.stamp=self.ground_stamp=self.scooter_stamp=-1e9
            return []
        if f.get('ego',{}).get('camera_shock'):
            self.grounding=[];self.ground_stamp=-1e9
            self.scooters=[];self.scooter_stamp=-1e9
        for p in self.public:p['box']=warp_dense(p['box'],backflow,w,h) if backflow is not None and not f.get('ego',{}).get('camera_shock') else warp(p['box'],f.get('matrix'))
        if f.get('public') is not None:self.public=copy.deepcopy(f['public']);self.stamp=t
        ps=self.public if t-self.stamp<=.65 else []
        for g in self.grounding:g['box']=warp_dense(g['box'],backflow,w,h) if backflow is not None else warp(g['box'],f.get('matrix'))
        if f.get('grounding') is not None:self.grounding=copy.deepcopy(f['grounding']);self.ground_stamp=t
        for g in self.scooters:g['box']=warp_dense(g['box'],backflow,w,h) if backflow is not None else warp(g['box'],f.get('matrix'))
        if f.get('grounding_scooters') is not None:self.scooters=copy.deepcopy(f['grounding_scooters']);self.scooter_stamp=t
        ds=[]
        for raw in f['detections']:
            d=dict(raw,source='custom');cid=d['class_id']
            if d['confidence']<(.08 if cid==9 else .18):continue
            semantics=scene.sample(d['box'],w,h);d.update(semantics)
            support=sorted([(iou(d['box'],p['box']),p) for p in ps],key=lambda x:x[0],reverse=True)
            if support and support[0][0]>=.4:
                overlap,p=support[0];pc=p['coco_class']
                if pc==16 and p['confidence']>=.35 and cid in (0,1,2,3):
                    d['veto']='animal_misclassified';ds.append(d);continue
                if pc in COCO and p['confidence']>=.35:
                    mapped=COCO[pc]
                    if (mapped==0 and (cid==0 or semantics.get('body_person_fraction',0)>.2)) or cid in (1,2,3) and mapped in (1,3):
                        d.update(original_class=d['class'],class_id=mapped,**{'class':NAMES[mapped]})
                    d['public_support']=p['confidence'];d['public_class']=pc
            cid=d['class_id']
            if cid in (1,2,3,4,5,6) and semantics.get('body_person_fraction',0)>.65 and semantics.get('body_vehicle_fraction',0)<.03:
                d.update(original_class=d['class'],class_id=0,**{'class':'person'});cid=0
            if cid in (0,1,2,3,4,5,6) and d.get('public_support',0)<.5 and semantics.get('body_solid_fraction',0)>.45 and semantics.get('body_vehicle_fraction',0)<.05 and semantics.get('body_person_fraction',0)<.05:
                d.update(original_class=d['class'],class_id=7,**{'class':'obstacle'});cid=7
            if cid==0 and semantics.get('body_ground_fraction',0)>.65 and semantics.get('body_person_fraction',0)<.04:
                d['veto']='ground_shadow'
            if cid!=9 and semantics.get('body_animal_fraction',0)>.12:d['veto']='animal_misclassified'
            ds.append(d)
        for p in ps:
            pc=p['coco_class']
            if pc not in COCO or p['confidence']<(.15 if pc==9 else .35):continue
            cid=COCO[pc]
            # Never count transported public colors as new observations.
            if cid==9 and ('color_features' not in p or abs(p.get('color_time_s',-1e9)-t)>1e-6):continue
            if cid==9 and any(d['class_id']==9 and iou(d['box'],p['box'])>.2 for d in ds):continue
            if any(iou(d['box'],p['box'])>.4 for d in ds):continue
            d=dict(box=p['box'],confidence=p['confidence'],class_id=cid,**{'class':NAMES[cid]},source='public',public_support=p['confidence'],public_class=pc)
            if cid==9:d['color_features']=p['color_features']
            d.update(scene.sample(d['box'],w,h));ds.append(d)
            if cid==0 and d.get('body_animal_fraction',0)>.12:d['veto']='animal_misclassified'
            if cid==0 and d.get('body_ground_fraction',0)>.65 and d.get('body_person_fraction',0)<.04:d['veto']='ground_shadow'
        for d in scene.proposals(w,h):
            if any(same_object(d['box'],x['box']) and not separate_post_and_vehicle(d,x,h) for x in ds if x['class_id']!=9 and not x.get('veto')):continue
            d.update(scene.sample(d['box'],w,h));ds.append(d)
        if t-self.ground_stamp<=2.3:
            for g in self.grounding:
                if any(same_object(g['box'],d['box']) and d['confidence']>=.4 and not d.get('veto') for d in ds if d['class_id']!=9):continue
                b=g['box'];label=g['label']
                aspect=(b[3]-b[1])/max(1,b[2]-b[0])
                if aspect<2. and not any(word in label for word in ('bench','trash')) and not (aspect<1/3 and g['confidence']>=.45):continue
                kind='horizontal_barrier' if 'barrier' in label and (b[2]-b[0])>3*(b[3]-b[1]) else 'grounded_object'
                d=dict(box=b,confidence=g['confidence'],class_id=7,**{'class':'obstacle'},source='grounding',grounding_label=label,
                       semantic_kind=kind,grounding_time=self.ground_stamp)
                d.update(scene.sample(b,w,h));ds.append(d)
        if t-self.scooter_stamp<=2.3:
            for g in self.scooters:
                if 'scooter' not in g['label'] or g['confidence']<.50:continue
                x1,y1,x2,y2=g['box'];bw=x2-x1;bh=y2-y1
                # A rescue prompt also returns poles, barrier arms and cropped
                # railings. Do not let these overwrite an obstacle hypothesis.
                if bw>2.2*bh or y1<.04*h and bh>.6*h:continue
                if y2>.97*h and (x1<.015*w or x2>.985*w):continue
                sem=scene.sample(g['box'],w,h)
                if sem.get('body_person_fraction',0)>.3 or sem.get('body_animal_fraction',0)>.1:continue
                candidate=dict(g,class_id=2)
                overlap=[d for d in ds if (same_object(g['box'],d['box']) or iou(g['box'],d['box'])>.4) and not separate_post_and_vehicle(candidate,d,h)]
                if any(d['class_id'] in (0,1,2,3,4,5,6) and d['confidence']>=.4 and not d.get('veto') for d in overlap):continue
                ds=[d for d in ds if not any(d is x for x in overlap)]
                ds.append(dict(box=g['box'],confidence=g['confidence'],class_id=2,**{'class':'scooter'},source='grounding_scooter',
                    grounding_label=g['label'],grounding_time=self.scooter_stamp,**sem))
        # Duplicate classes from the same detector can otherwise warn twice.
        # A bench/table can switch between whole-object and individual-seat
        # detections. A connected bench segment supplies a common instance box.
        for d in ds:
            if d['class_id']==7 and d.get('bench_instance_box'):
                d['detector_box']=d['box'];d['box']=d['bench_instance_box']
        out=[]
        def rank(d):
            model=d['source'] in ('custom','public') and d['confidence']>=.35 and not d.get('veto')
            return (2 if model else 1 if d['source'] in ('surface','grounding','grounding_scooter') else 0,d['confidence']+.35*(d['source']=='public' and d['class_id']==7))
        for d in sorted(ds,key=rank,reverse=True):
            d['box']=np.clip(d['box'],[0,0,0,0],[w,h,w,h]).tolist()
            x1,y1,x2,y2=d['box']
            identity=d.get('identity_component')
            if identity and not d.get('veto') and any(
                    x.get('identity_component')==identity and not x.get('veto')
                    and (d['class_id']==x['class_id']==7 or d['class_id'] in (1,2,3) and x['class_id'] in (1,2,3))
                    for x in out):continue
            if self.suppress_wearer and d['class_id']==0 and y2>.92*h and (x2-x1)>.4*w and (y2-y1)/max(1,x2-x1)<1.2:d['veto']='wearer_body_fragment'
            if self.suppress_wearer and d['class_id']!=9 and y2>=.94*h and y1>.25*h and (x2-x1)>.4*w and (y2-y1)/max(1,x2-x1)<1.4 and d.get('body_person_fraction',0)>.15:d['veto']='wearer_body_fragment'
            if self.suppress_wearer and d['class_id']==0 and y2>=.90*h and y1>.52*h and (x1<.03*w or x2>.97*w):d['veto']='wearer_body_fragment'
            if d['class_id'] in (0,1,2,3,4,5,6) and d.get('body_flat_fraction',0)>.85 and d.get('body_person_fraction',0)<.02 and d.get('body_vehicle_fraction',0)<.02:
                d['veto']='flat_ground_detection'
            vertical=next((x for x in out if d['class_id']==x['class_id']==7 and same_vertical_object(d['box'],x['box'])),None)
            if vertical is not None:
                if np.prod(np.subtract(d['box'][2:],d['box'][:2]))>np.prod(np.subtract(vertical['box'][2:],vertical['box'][:2])):
                    out.remove(vertical)
                else:continue
            if any(d['class_id']!=9 and x['class_id']!=9 and same_object(d['box'],x['box']) and not separate_post_and_vehicle(d,x,h) for x in out):continue
            out.append(d)
        return out

class ObjectTracker:
    def __init__(self):self.tracks={};self.next_id=1
    def step(self,f,ds,backflow=None):
        t=f['time_s'];out=[]
        for tid,s in list(self.tracks.items()):
            if t-s['last']>3.:del self.tracks[tid];continue
            s['box']=warp_dense(s['box'],backflow,f['width'],f['height']) if backflow is not None and s['class_id']!=9 and not f.get('ego',{}).get('camera_shock') else warp(s['box'],f.get('matrix'))
        ids=list(self.tracks);cost=np.full((len(ids),len(ds)),1e3)
        for i,tid in enumerate(ids):
            s=self.tracks[tid]
            for j,d in enumerate(ds):
                if (s['class_id']==9)!=(d['class_id']==9):continue
                a=np.asarray(s['box']);b=np.asarray(d['box']);overlap=iou(a,b)
                ga=0 if s['class_id']==0 else 1 if s['class_id'] in (1,2,3) else 2 if s['class_id'] in (4,5,6) else 3
                gb=0 if d['class_id']==0 else 1 if d['class_id'] in (1,2,3) else 2 if d['class_id'] in (4,5,6) else 3
                if ga!=gb and overlap<(.7 if 0 in (ga,gb) else .55):continue
                delta=np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)
                size=max(15.,min(np.linalg.norm(a[2:]-a[:2]),np.linalg.norm(b[2:]-b[:2])))
                light_pan=s['class_id']==9 and f.get('camera',{}).get('method')=='orb_fallback' and t-s['last']<=.5
                if overlap<.08 and delta>(1.8 if light_pan else .3)*size:continue
                cost[i,j]=1-overlap+.15*delta/size
                if light_pan:cost[i,j]-=.3
        matches={}
        if cost.size:
            ii,jj=linear_sum_assignment(cost)
            matches={j:ids[i] for i,j in zip(ii,jj) if cost[i,j]<1.1}
        for j,d in enumerate(ds):
            if j not in matches:
                tid=self.next_id;self.next_id+=1
                s=dict(box=d['box'],class_id=d['class_id'],first=t,last=t,count=0,peak=0.,votes=np.zeros(10))
                self.tracks[tid]=s
            else:tid=matches[j];s=self.tracks[tid]
            s['last']=t;s['box']=d['box'];s['count']+=1;s['peak']=max(s['peak'],d['confidence'])
            recent=s.setdefault('confidence_history',[])
            recent.append((t,d['confidence']))
            recent[:]=[(stamp,conf) for stamp,conf in recent if t-stamp<=1.5]
            if d.get('grounding_time') is not None and d['grounding_time']!=s.get('grounding_time'):
                s['grounding_time']=d['grounding_time'];s['grounding_confirmations']=s.get('grounding_confirmations',0)+1
            if d.get('veto')=='animal_misclassified':s['animal_until']=t+3.
            if t<s.get('animal_until',-1e9):d=dict(d,veto='animal_misclassified')
            s['votes']*=.7;s['votes'][d['class_id']]+=d['confidence']*(1.6 if d.get('public_class')==0 else 1.)
            cid=int(s['votes'].argmax());s['class_id']=cid
            out.append(dict(d,track_id=tid,class_id=cid,**{'class':NAMES[cid]},track_age=t-s['first'],track_count=s['count'],peak_confidence=s['peak'],recent_peak_confidence=max(conf for _,conf in recent),grounding_confirmations=s.get('grounding_confirmations',0),order=j))
        return out
