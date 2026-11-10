"""Background affine motion with descriptor fallback for abrupt camera pans."""
import cv2
import numpy as np
from camera import estimate_camera
from ego_motion import FlowRecorder

class ImageMotion:
    def __init__(self):
        self.flow=FlowRecorder();self.previous=None;self.previous_mask=None
        self.orb=cv2.ORB_create(nfeatures=1800,fastThreshold=12)
    def step(self,img,boxes=()):
        h,w=img.shape[:2];s=min(1,640/max(w,h));shape=(round(w*s),round(h*s))
        gray=cv2.cvtColor(cv2.resize(img,shape),cv2.COLOR_BGR2GRAY);mask=np.full(gray.shape,255,np.uint8)
        for b in boxes:
            x1,y1,x2,y2=np.rint(np.asarray(b)*s).astype(int)
            mask[max(0,y1-3):min(shape[1],y2+4),max(0,x1-3):min(shape[0],x2+4)]=0
        pairs=self.flow.step(img,boxes);matrix,info=estimate_camera(pairs,w,h,min_ratio=.35)
        if matrix is None and self.previous is not None and self.previous.shape==gray.shape:
            k1,d1=self.orb.detectAndCompute(self.previous,self.previous_mask);k2,d2=self.orb.detectAndCompute(gray,mask)
            if d1 is not None and d2 is not None and len(d1)>25 and len(d2)>25:
                matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(d1,d2,k=2)
                matches=[a for pair in matches if len(pair)==2 for a,b in [pair] if a.distance<.75*b.distance]
                if len(matches)>=20:
                    p=np.float32([k1[m.queryIdx].pt for m in matches]);q=np.float32([k2[m.trainIdx].pt for m in matches])
                    m,good=cv2.estimateAffinePartial2D(p,q,method=cv2.RANSAC,ransacReprojThreshold=3,maxIters=2000)
                    if m is not None and good is not None:
                        keep=good.ravel()>0;n=int(keep.sum());ratio=n/len(matches);scale=np.hypot(m[0,0],m[1,0]);spread=np.ptp(p[keep],axis=0)
                        if n>=20 and ratio>=.4 and .75<scale<1.3 and np.all(spread>np.array(shape)*.15):
                            matrix=m.copy();matrix[:,2]/=s
                            info=dict(camera_ok=True,inliers=n,inlier_ratio=ratio,method='orb_fallback')
        self.previous,self.previous_mask=gray,mask
        return matrix,info,pairs
