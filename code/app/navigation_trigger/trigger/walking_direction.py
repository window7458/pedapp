"""Separate camera rotation from translation using static multi-frame features.

Uncalibrated focal length is an approximation. Translation magnitude is NOT
metric speed. Degenerate/low-parallax estimates explicitly remain unknown.
OpenCV recoverPose uses X2=R X1+t; camera displacement in view 2 is -t.
"""
from collections import deque
import math
import cv2
import numpy as np

def intrinsic(w,h,focal=None):
    f=float(focal or 26/36*max(w,h))
    return np.array([[f,0,w/2],[0,f,h/2],[0,0,1.]],np.float64)

def recover_direction(p,q,k):
    if len(p)<30:return None
    e,mask=cv2.findEssentialMat(p,q,k,method=cv2.RANSAC,prob=.999,threshold=1.,maxIters=700)
    if e is None or mask is None:return None
    best=None
    for block in np.split(e,len(e)//3):
        n,r,t,good=cv2.recoverPose(block,p,q,k,mask=mask.copy())
        good=good.ravel()>0
        if n<25 or n/len(p)<.45:continue
        rays=np.c_[p,np.ones(len(p))]@np.linalg.inv(k).T
        rotated=rays@r.T;proj=rotated@k.T;proj=proj[:,:2]/proj[:,2:]
        parallax=float(np.median(np.linalg.norm(q[good]-proj[good],axis=1)))
        v=-t[:,0]
        if parallax<.65 or v[2]<.12 or abs(v[1])>.85:continue
        heading=math.atan2(v[0],v[2])
        if abs(heading)>math.radians(80):continue
        score=n/len(p)*min(1,parallax/3.)
        value=dict(heading_rad=heading,rotation=r,direction=v,inliers=int(n),
                   inlier_ratio=n/len(p),parallax_px=parallax,confidence=score)
        if best is None or score>best['confidence']:best=value
    return best

class WalkingDirection:
    def __init__(self):
        self.reference=None;self.samples=deque();self.last_good=-1e9
        self.heading=0.;self.vertical=0.;self.still_since=None;self.last=None
    def step(self,img,time_s,boxes=()):
        h,w=img.shape[:2];s=min(1.,640/max(h,w));sw,sh=round(w*s),round(h*s)
        gray=cv2.cvtColor(cv2.resize(img,(sw,sh)),cv2.COLOR_BGR2GRAY)
        mask=np.full(gray.shape,255,np.uint8)
        # Never infer walking direction from moving people/vehicles/guide dogs.
        for b in boxes:
            x1,y1,x2,y2=np.asarray(b)*s
            mask[max(0,int(y1)-4):min(sh,int(y2)+5),max(0,int(x1)-4):min(sw,int(x2)+5)]=0
        result=dict(heading_rad=self.heading,valid=False,observed=False,
                    confidence=0.,inliers=0,parallax_px=None,stationary=False)
        if self.last is not None and time_s<=self.last:raise ValueError('Nonmonotonic time')
        if self.last is not None and time_s-self.last>1.:
            self.reference=None;self.samples.clear();self.last_good=-1e9
        self.last=time_s
        if self.reference is not None:
            old,oldmask,stamp=self.reference
            if gray.shape==old.shape and time_s-stamp>=.38:
                p=cv2.goodFeaturesToTrack(old,650,.008,7,mask=oldmask,blockSize=7)
                if p is not None:
                    q,ok,_=cv2.calcOpticalFlowPyrLK(old,gray,p,None,winSize=(25,25),maxLevel=4)
                    if q is not None:
                        back,bok,_=cv2.calcOpticalFlowPyrLK(gray,old,q,None,winSize=(25,25),maxLevel=4)
                        if back is not None:
                            p=p.reshape(-1,2);q=q.reshape(-1,2)
                            good=ok.ravel().astype(bool)&bok.ravel().astype(bool)&(np.linalg.norm(back.reshape(-1,2)-p,axis=1)<1.)
                            good&=np.isfinite(q).all(1)&(q[:,0]>=0)&(q[:,0]<sw)&(q[:,1]>=0)&(q[:,1]<sh)
                            ij=np.clip(q,[0,0],[sw-1,sh-1]).astype(int);good&=mask[ij[:,1],ij[:,0]]>0
                            p=p[good];q=q[good]
                            motion=recover_direction(p,q,intrinsic(sw,sh)) if len(p)>=30 else None
                            # Background at rest is positive evidence; invalid geometry
                            # alone must never label the pedestrian stationary.
                            still=len(p)>=30 and float(np.median(np.linalg.norm(q-p,axis=1)))<.7
                            if still:
                                if self.still_since is None:self.still_since=stamp
                            else:self.still_since=None
                            result['stationary']=self.still_since is not None and time_s-self.still_since>=.8
                            if motion:
                                self.samples.append((time_s,motion['heading_rad'],motion['direction'][1]/motion['direction'][2]))
                                result.update({k:v for k,v in motion.items() if k not in ('rotation','direction')});result['observed']=True
                if time_s-stamp>=.58:self.reference=(gray,mask,time_s)
        else:self.reference=(gray,mask,time_s)
        while self.samples and time_s-self.samples[0][0]>1.4:self.samples.popleft()
        if len(self.samples)>=2 and result['observed']:
            angles=np.array([v[1] for v in self.samples]);med=float(np.median(angles))
            # Consistency over multiple past pairs, not one RANSAC solution.
            if np.median(np.abs(angles-med))<math.radians(12):
                self.heading=.65*self.heading+.35*med;self.vertical=float(np.median([v[2] for v in self.samples]));self.last_good=self.samples[-1][0]
        valid=time_s-self.last_good<=2.
        result.update(heading_rad=self.heading,vertical_slope=self.vertical,valid=valid,
                      age_s=None if not valid else time_s-self.last_good,
                      camera_motion_angle_deg=math.degrees(self.heading))
        return result
