"""Causal walking-speed and path estimation derived from Park's trigger_hitbox.

Ground distances assume a 1.70 m camera, a flat floor, and a 26 mm equivalent
lens. Estimates are accompanied by validity flags, never called measurements.
"""
import math
import cv2
import numpy as np

CAM_H=1.70
def focus_of_expansion(matrix):
    if matrix is None:return None
    a=np.asarray(matrix)[:,:2];shift=np.asarray(matrix)[:,2]
    scale=math.sqrt(abs(np.linalg.det(a)))
    if scale<1.002 or abs(math.atan2(a[1,0],a[0,0]))>math.radians(2):return None
    b=np.eye(2)-a
    if np.linalg.svd(b,compute_uv=False)[-1]<1e-3:return None
    # For p'=A p+t, a stationary expansion centre satisfies (I-A)c=t.
    return np.linalg.solve(b,shift)

def walking_speed(pairs,f,horizon,w,h,dt):
    if pairs is None or len(pairs)<15 or dt<=0:return None
    x,y,ny=pairs[:,0],pairs[:,1],pairs[:,3]
    keep=(np.abs(x-w/2)<.25*w)&(y>horizon+.12*h)
    if keep.sum()<15:return None
    z=(y[keep]-horizon)**2
    design=np.column_stack([np.ones(len(z)),z])
    dy=ny[keep]-y[keep]
    coef=np.linalg.lstsq(design,dy,rcond=None)[0]
    residual=dy-design@coef
    mad=np.median(np.abs(residual-np.median(residual)))
    good=np.abs(residual-np.median(residual))<=max(1.5,3*mad)
    if good.sum()<15:return None
    coef=np.linalg.lstsq(design[good],dy[good],rcond=None)[0]
    speed=float(coef[1]*f*CAM_H/dt)
    return max(0.,speed) if -.5<speed<4. else None

class FlowRecorder:
    def __init__(self):self.gray=None;self.mask=None
    def step(self,img,boxes):
        h,w=img.shape[:2];s=min(1.,640/max(h,w));sw,sh=round(w*s),round(h*s)
        scale=np.array([sw/w,sh/h]);gray=cv2.cvtColor(cv2.resize(img,(sw,sh)),cv2.COLOR_BGR2GRAY)
        mask=np.full((sh,sw),255,np.uint8)
        for box in boxes:
            x1,y1,x2,y2=np.array(box)*np.tile(scale,2)
            mask[max(0,int(y1)-5):min(sh,int(np.ceil(y2))+5),max(0,int(x1)-5):min(sw,int(np.ceil(x2))+5)]=0
        pairs=np.empty((0,4),np.float32)
        if self.gray is not None and self.gray.shape==gray.shape:
            p=cv2.goodFeaturesToTrack(self.gray,500,.01,10,mask=self.mask,blockSize=7)
            if p is not None and len(p)>=15:
                q,ok,_=cv2.calcOpticalFlowPyrLK(self.gray,gray,p,None,winSize=(21,21),maxLevel=3)
                if q is not None:
                    back,bok,_=cv2.calcOpticalFlowPyrLK(gray,self.gray,q,None,winSize=(21,21),maxLevel=3)
                    if back is not None:
                        p=p.reshape(-1,2);q=q.reshape(-1,2);back=back.reshape(-1,2)
                        keep=ok.ravel().astype(bool)&bok.ravel().astype(bool)&(np.linalg.norm(back-p,axis=1)<=1.5)
                        keep&=np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<sw)&(q[:,1]>=0)&(q[:,1]<sh)
                        ix=np.floor(np.clip(q,[0,0],[sw-1,sh-1])).astype(int)
                        keep&=mask[ix[:,1],ix[:,0]]>0
                        pairs=np.c_[p[keep]/scale,q[keep]/scale].astype(np.float32)
        self.gray,self.mask=gray,mask
        return pairs

class EgoMotion:
    def __init__(self):
        self.last=None;self.focus=None;self.walk=1.2;self.last_valid=-1e9
        self.shake=np.zeros(2);self.smooth=np.zeros(2)
    def step(self,frame,pairs):
        t,w,h=frame['time_s'],frame['width'],frame['height'];dt=t-self.last if self.last is not None else .1
        if dt<=0:raise ValueError('Timestamps must increase')
        self.last=t
        if self.focus is None:self.focus=np.array([w/2,h/2.])
        m=np.array(frame['matrix']) if frame.get('matrix') is not None else None
        shock=False
        if m is not None:
            center=np.array([w/2,h/2]);move=m[:,:2]@center+m[:,2]-center
            shock=bool(np.linalg.norm(move)/min(w,h)>.12 or abs(math.atan2(m[1,0],m[0,0]))>math.radians(12))
            candidate=focus_of_expansion(m)
            if candidate is not None and 0<=candidate[0]<w and 0<=candidate[1]<h:
                alpha=1-math.exp(-dt/2.5);self.focus=(1-alpha)*self.focus+alpha*candidate
            self.shake+=move
        a=1-math.exp(-dt/.65);self.smooth=(1-a)*self.smooth+a*self.shake
        shift=self.shake-self.smooth
        cx=float(np.clip(self.focus[0],.4*w,.6*w)+np.clip(shift[0],-.08*w,.08*w))
        # Head/body cameras may point down: forcing a 40%-high horizon makes
        # nearby feet appear tens of metres away. Retain a bounded observed FOE.
        horizon=float(np.clip(self.focus[1],.2*h,.7*h)+np.clip(shift[1],-.08*h,.08*h))
        f=26/36*max(w,h);v=None if shock else walking_speed(pairs,f,horizon,w,h,dt)
        if v is not None:
            a=1-math.exp(-dt/.8);self.walk=(1-a)*self.walk+a*v;self.last_valid=t
        valid=t-self.last_valid<=2.
        if not valid:self.walk=1.2
        return dict(time_s=t,frame=frame['frame'],focal_px=f,camera_cx=cx,horizon=horizon,
                    walk_mps=float(self.walk),walk_valid=valid,walk_observed=v is not None,
                    camera_shock=shock,flow_points=len(pairs))

class PathEstimator:
    """Use connected floor segments, then smooth their ground-plane centreline."""
    def __init__(self):self.heading=0.;self.last_good=-1e9
    def step(self,frame,ego,surface=None,age=None):
        t=frame['time_s'];w,h=frame['width'],frame['height'];us=0;certainty=0.;accepted=False
        if surface is not None and age is not None and age<=1.2:
            sh,sw=surface.shape;foot=surface[int(.88*sh):,int(.3*sw):int(.7*sw)]
            frac=np.bincount(foot.ravel(),minlength=7)/max(1,foot.size);us=int(frac.argmax());certainty=float(frac[us])
            walkable=(1,3) if us in (0,1,4,5,6) else (2,3)
            xs,zs=[],[]
            for row in range(int(.56*sh),int(.96*sh),3):
                yy=(row+.5)/sh*h;dy=yy-ego['horizon']
                if dy<40:continue
                z=ego['focal_px']*CAM_H/dy
                if not 2<z<15:continue
                good=np.isin(surface[row],walkable);edges=np.diff(np.r_[False,good,False].astype(int))
                runs=[(a,b) for a,b in zip(np.where(edges==1)[0],np.where(edges==-1)[0]) if b-a>=.08*sw]
                if not runs:continue
                pred=ego['camera_cx']/w*sw+math.tan(self.heading)*ego['focal_px']/w*sw
                lo,hi=min(runs,key=lambda ab:abs((ab[0]+ab[1])/2-pred))
                x=(((lo+hi)/2/sw*w)-ego['camera_cx'])*z/ego['focal_px']
                xs.append(x);zs.append(z)
            if certainty>=.55 and len(zs)>=4 and np.ptp(zs)>1.5:
                slope=float(np.polyfit(zs,xs,1)[0]);angle=float(np.clip(math.atan(slope),-math.pi/6,math.pi/6))
                self.heading=.8*self.heading+.2*angle;self.last_good=t;accepted=True
        if ego['camera_shock']:self.heading=0.;self.last_good=-1e9;accepted=False
        if t-self.last_good>2.:self.heading*=.9
        return dict(ego,path_heading_rad=self.heading,path_valid=t-self.last_good<=2.,path_observed=accepted,
                    user_surface=us,surface_confidence=certainty,surface_age_s=age)
