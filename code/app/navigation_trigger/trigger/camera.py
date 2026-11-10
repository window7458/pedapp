"""Same RANSAC tolerance in the 640-pixel optical-flow coordinate system."""
import cv2,numpy as np

def estimate_camera(pairs,width,height,min_ratio=.6):
    matrix=None;count=0;ratio=0.
    scale=min(1.,640/max(width,height))
    if len(pairs)>=20:
        small=pairs*scale
        candidate,inliers=cv2.estimateAffinePartial2D(np.ascontiguousarray(small[:,:2]),np.ascontiguousarray(small[:,2:]),method=cv2.RANSAC,ransacReprojThreshold=2,maxIters=2000)
        if candidate is not None and inliers is not None:
            count=int(inliers.sum());ratio=count/len(pairs);s=np.hypot(candidate[0,0],candidate[1,0])
            spread=np.ptp(small[inliers.ravel().astype(bool),:2],axis=0)
            if count>=30 and ratio>=min_ratio and .9<=s<=1.1 and spread[0]>=width*scale*.2 and spread[1]>=height*scale*.2:
                matrix=candidate.copy();matrix[:,2]/=scale
    return matrix,dict(camera_ok=matrix is not None,inliers=count,inlier_ratio=ratio)
