"""Causal semantic context and physical obstacle proposals.

Unknown pixels are not obstacles. Proposals require a recognized solid/vegetation
component with a visible ground interface, then tracking/approach confirmation.
"""
import math
import cv2
import numpy as np
WALK=(7,9,11,15);ROAD=(10,12,13,14);CROSS=(8,23);GROUND=WALK+ROAD+CROSS+(2,24)
SOLID=(5,33,34,38,39,40,42,44,45,46,47,51)
PEOPLE=(19,20,21,22)

class SceneContext:
    def __init__(self):self.raw=None;self.stamp=-1e9;self.bench_components=None;self.identity_components={}
    def step(self,f,new=None,backflow=None):
        t=f['time_s'];w,h=f['width'],f['height']
        if new is not None:self.raw=new.copy();self.stamp=t
        elif self.raw is not None:
            if f.get('ego',{}).get('camera_shock'):self.raw=None
            elif backflow is not None:
                sh,sw=self.raw.shape
                yy,xx=np.mgrid[:sh,:sw].astype(np.float32)
                self.raw=cv2.remap(self.raw,xx+backflow[:,:,0],yy+backflow[:,:,1],cv2.INTER_NEAREST,borderValue=255)
            elif f.get('matrix') is None or f.get('ego',{}).get('camera_shock'):self.raw=None
            else:
                sh,sw=self.raw.shape;s=np.diag([sw/w,sh/h,1.]);m=np.eye(3);m[:2]=f['matrix']
                self.raw=cv2.warpAffine(self.raw,(s@m@np.linalg.inv(s))[:2],(sw,sh),flags=cv2.INTER_NEAREST,borderValue=255)
        if t-self.stamp>1.3 or not f.get('image_quality',{}).get('valid',True):self.raw=None
        raw=self.raw;ego=f['ego'];direction=f.get('direction',{});focal=ego['focal_px']
        self.bench_components=None
        self.identity_components={}
        heading=direction.get('heading_rad',0.) if direction.get('valid') else 0.
        horizon=h/2+focal*direction.get('vertical_slope',0.) if direction.get('valid') else ego['horizon']
        floor_limit=None
        if raw is not None:
            sh,sw=raw.shape;floor=np.isin(raw,GROUND).astype(np.uint8)
            n,cc,stats,_=cv2.connectedComponentsWithStats(floor,8)
            candidates=[i for i in range(1,n) if stats[i,cv2.CC_STAT_TOP]+stats[i,cv2.CC_STAT_HEIGHT]>.8*sh and stats[i,cv2.CC_STAT_AREA]>.04*sh*sw]
            if candidates:
                region=np.isin(cc,candidates);tops=[]
                for col in range(int(.1*sw),int(.9*sw)):
                    ys=np.flatnonzero(region[:,col])
                    if len(ys)>.15*sh:tops.append(int(ys[0]))
                if len(tops)>.2*sw:
                    # Visible connected flat ground must lie below the horizon.
                    # This is an upper bound, not a measured horizon or a choice
                    # of walking direction; it also works on a broad plaza.
                    floor_limit=float(np.percentile(tops,25)/sh*h-.025*h)
                    horizon=min(horizon,floor_limit)
        horizon=float(np.clip(horizon,.12*h,.65*h))
        ctx=dict(ego,path_heading_rad=heading,path_valid=bool(direction.get('valid')),heading_source='visual_translation' if direction.get('valid') else 'unknown',
                 horizon=horizon,camera_cx=w/2,user_surface=0,surface_confidence=0.,surface_age_s=None if raw is None else t-self.stamp,
                 direction=direction,stationary=bool(direction.get('stationary')),floor_horizon_upper_bound=floor_limit)
        if raw is not None:
            sh,sw=raw.shape
            foot_cx=(w/2+focal*math.tan(heading))/w*sw
            foot_cx=float(np.clip(foot_cx,.2*sw,.8*sw))
            foot=raw[int(.88*sh):int(.99*sh),max(0,int(foot_cx-.25*sw)):min(sw,int(foot_cx+.25*sw))]
            # Restrict voting to ground; guide dog/body pixels do not become road.
            counts=np.array([np.isin(foot,v).sum() for v in (WALK,ROAD,CROSS)])
            if counts.sum()>=foot.size*.35:
                ctx['user_surface']=int(counts.argmax())+1;ctx['surface_confidence']=float(counts.max()/counts.sum())
        return ctx
    def sample(self,box,w,h):
        if self.raw is None:return dict(surface_known=False)
        raw=self.raw;sh,sw=raw.shape
        x1,y1,x2,y2=np.clip(np.asarray(box)*[sw/w,sh/h,sw/w,sh/h],[0,0,0,0],[sw,sh,sw,sh])
        if x2<=x1 or y2<=y1:return dict(surface_known=False)
        def patch(a,b,c,d):return raw[max(0,int(b)):min(sh,int(math.ceil(d))),max(0,int(a)):min(sw,int(math.ceil(c)))]
        body=patch(x1+(x2-x1)*.15,y1+(y2-y1)*.15,x2-(x2-x1)*.15,y2)
        foot=patch(x1+(x2-x1)*.15,y2-1,x2-(x2-x1)*.15,y2+max(3,.025*sh))
        if not body.size or not foot.size:return dict(surface_known=False)
        contact=None;vehicle_foot=None
        ix1,ix2=max(0,int(x1)),min(sw,int(math.ceil(x2)));iy1,iy2=max(0,int(y1+.5*(y2-y1))),min(sh,int(math.ceil(y2)))
        lower=raw[iy1:iy2,ix1:ix2];vy,vx=np.where(np.isin(lower,(54,55,56,59,60,61)))
        if len(vy)>=8:
            bottom=int(np.percentile(vy,98));near=vy>=bottom-max(1,int(.035*(y2-y1)))
            if near.sum()>=3:
                ax,bx=np.percentile(vx[near],[5,95]);cy=iy1+bottom+1
                if bx-ax>=2:
                    contact=[float((ix1+ax)/sw*w),float(cy/sh*h),float((ix1+bx+1)/sw*w),float(cy/sh*h)]
                    vehicle_foot=patch(ix1+ax,cy-1,ix1+bx+1,cy+max(3,.025*sh))
        road=float(np.isin(foot,ROAD+CROSS).mean());walk=float(np.isin(foot,WALK).mean());support=road+walk
        result=dict(surface_known=True,road_fraction=road/max(.25,support),
                    sidewalk_fraction=walk/max(.25,support),foot_ground_support=support,vehicle_contact_span=contact,
                    body_ground_fraction=float(np.isin(body,GROUND+(26,28,29,30)).mean()),
                    body_flat_fraction=float(np.isin(body,GROUND+(26,28,29)).mean()),
                    body_person_fraction=float(np.isin(body,PEOPLE).mean()),
                    body_solid_fraction=float(np.isin(body,SOLID).mean()),
                    body_boundary_fraction=float(np.isin(body,(3,4,5)).mean()),
                    body_vegetation_fraction=float((body==30).mean()),
                    body_vehicle_fraction=float(np.isin(body,(52,54,55,56,57,59,60,61)).mean()),
                    body_animal_fraction=float((body==1).mean()))
        if (body==33).mean()>.25:
            if self.bench_components is None:
                self.bench_components=cv2.connectedComponentsWithStats((raw==33).astype(np.uint8),8)
            _,components,stats,_=self.bench_components
            region=components[max(0,int(y1)):min(sh,int(y2)+1),max(0,int(x1)):min(sw,int(x2)+1)]
            counts=np.bincount(region.ravel(),minlength=1);counts[0]=0
            if len(counts)>1 and counts.max()>=8:
                bx,by,bw,bh,area=stats[counts.argmax()]
                if area<.35*sh*sw:
                    result['bench_instance_box']=[float(bx/sw*w),float(by/sh*h),float((bx+bw)/sw*w),float((by+bh)/sh*h)]
        if vehicle_foot is not None and vehicle_foot.size:
            vr=float(np.isin(vehicle_foot,ROAD+CROSS).mean());vw=float(np.isin(vehicle_foot,WALK).mean());vs=vr+vw
            result.update(vehicle_road_fraction=vr/max(.25,vs),vehicle_sidewalk_fraction=vw/max(.25,vs))
        # Semantic components help join duplicate part boxes. They are only an
        # identity cue, never an independent risk/near-distance cue.
        if result['body_solid_fraction']>.35 and np.isin(body,(44,45,46,47)).mean()>.35:
            key='pole';mask=np.isin(raw,(44,45,46,47)).astype(np.uint8)
        else:key=None
        if key is not None:
            if key not in self.identity_components:
                self.identity_components[key]=cv2.connectedComponentsWithStats(mask,8)
            _,components,stats,_=self.identity_components[key]
            region=components[max(0,int(y1)):min(sh,int(y2)+1),max(0,int(x1)):min(sw,int(x2)+1)]
            counts=np.bincount(region.ravel(),minlength=1);counts[0]=0
            if len(counts)>1 and counts.max()>=8:
                component=int(counts.argmax());bx,by,bw,bh,area=stats[component]
                narrow=bw<.12*sw and bh/max(1,bw)>2.5
                if narrow and area>.40*bw*bh and bh>.08*sh:
                    result.update(identity_component=f'{key}:{component}',
                                  identity_box=[float(bx/sw*w),float(by/sh*h),float((bx+bw)/sw*w),float((by+bh)/sh*h)])
        return result
    def proposals(self,w,h):
        if self.raw is None:return []
        raw=self.raw;sh,sw=raw.shape;out=[]
        floor=np.isin(raw,GROUND).astype(np.uint8)
        # Separate thin solid objects from broad vegetation. Vegetation is only
        # eligible where a narrow lower stem interrupts otherwise walkable floor.
        for kind,mask in [('solid',np.isin(raw,tuple(i for i in SOLID if i not in (5,34)))),('vegetation',raw==30)]:
            binary=mask.astype(np.uint8)
            if kind=='vegetation':binary[:int(.42*sh)]=0
            n,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
            for i in range(1,n):
                x,y,bw,bh,area=stats[i];bottom=y+bh
                if area<8 or bh<.05*sh or bottom<.43*sh or bw>.65*sw:continue
                # Reject grass verges/walls: a physical obstruction has floor
                # below it or on both sides of its lower portion.
                lo=max(y,bottom-max(3,int(.08*sh)));pad=max(2,int(.03*sw))
                left=floor[lo:bottom,max(0,x-pad):x];right=floor[lo:bottom,x+bw:min(sw,x+bw+pad)]
                below=floor[bottom:min(sh,bottom+max(3,int(.04*sh))),x:x+bw]
                fl=float(left.mean()) if left.size else 0.;fr=float(right.mean()) if right.size else 0.;fb=float(below.mean()) if below.size else 0.
                if kind=='vegetation':
                    if bw>.22*sw or bh/max(1,bw)<1.3 or min(fl,fr)<.35:continue
                elif max(fb,min(fl,fr))<.3:continue
                out.append(dict(box=[x/sw*w,y/sh*h,(x+bw)/sw*w,bottom/sh*h],class_id=7,**{'class':'obstacle'},confidence=.65,
                                source='surface',semantic_kind=kind,ground_interface=max(fb,min(fl,fr))))
        # A railing is a continuous boundary, but its protruding terminal post
        # can block a crossing. Isolate lower-contour protrusions rather than
        # promoting the whole fence or every upright bar to an obstacle.
        rail=np.isin(raw,(3,4,5)).astype(np.uint8)
        n,labels,stats,_=cv2.connectedComponentsWithStats(rail,8)
        for i in range(1,n):
            x,y,bw,bh,area=stats[i]
            if bw<.12*sw or bh<.15*sh or area<30:continue
            mask=labels==i;cols=np.flatnonzero(mask.any(0))
            bottoms=np.array([np.flatnonzero(mask[:,c])[-1]+1 for c in cols])
            cut=int(max(np.median(bottoms)+max(4,.035*sh),np.percentile(bottoms,80)))
            tail=mask.copy();tail[:cut]=False
            count,parts,boxes,_=cv2.connectedComponentsWithStats(tail.astype(np.uint8),8)
            for j in range(1,count):
                px,py,pw,ph,pa=boxes[j];bottom=py+ph
                if ph<.05*sh or not .012*sw<=pw<=.09*sw or bottom<.55*sh:continue
                if px<=1 or px+pw>=sw-1:continue
                yy=np.flatnonzero(mask[:,px:px+pw].any(1));top=int(yy[0])
                if bottom-top<.15*sh:continue
                below=floor[bottom:min(sh,bottom+max(3,int(.035*sh))),px:px+pw]
                if not below.size or below.mean()<.5:continue
                out.append(dict(box=[px/sw*w,top/sh*h,(px+pw)/sw*w,bottom/sh*h],class_id=7,**{'class':'obstacle'},confidence=.65,
                                source='surface',semantic_kind='guardrail_terminal_post',ground_interface=float(below.mean())))
        return out

class SurfaceFlow:
    """Backward dense flow transports a past semantic map to the current frame."""
    def __init__(self):self.previous=None
    def step(self,image):
        h,w=image.shape[:2];shape=(round(w*384/max(w,h)),round(h*384/max(w,h)))
        gray=cv2.cvtColor(cv2.resize(image,shape),cv2.COLOR_BGR2GRAY);flow=None
        if self.previous is not None and self.previous.shape==gray.shape:
            flow=cv2.calcOpticalFlowFarneback(gray,self.previous,None,.5,3,19,3,5,1.1,0)
        self.previous=gray
        return flow
