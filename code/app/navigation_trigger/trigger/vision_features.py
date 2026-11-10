"""Image-only traffic-light color measurements."""
import cv2
import numpy as np

def color_features(frame,box):
 h,w=frame.shape[:2];x1,y1,x2,y2=map(int,box)
 x1=max(0,min(x1,w-1));x2=max(x1+1,min(x2,w));y1=max(0,min(y1,h-1));y2=max(y1+1,min(y2,h))
 roi=frame[y1:y2,x1:x2]
 features={}
 for trim in [0,.1]:
  rh,rw=roi.shape[:2];dx=int(rw*trim);dy=int(rh*trim*.5)
  inner=roi[dy:rh-dy,dx:rw-dx];b,g,r=np.moveaxis(inner.astype(np.float32),-1,0)
  hh,s,v=np.moveaxis(cv2.cvtColor(inner,cv2.COLOR_BGR2HSV),-1,0)
  for value in [60,100,140]:
   for mode in ['hsv','rgb']:
    if mode=='hsv':
     red=((hh<12)|(hh>168))&(s>60)&(v>=value)
     green=(hh>32)&(hh<100)&(s>45)&(v>=value)
    else:
     red=(r>=value)&(r>g*1.5)&(r>b*1.5)
     green=(g>=value)&(g>r*1.3)&(g>b*.8)
    features[f'{mode}_{value}_{trim}']=[int(red.sum()),int(green.sum()),int(red.size)]
   if trim==.1:
    red=(r>=value)&(r>g*1.8)&(r>b*1.8)
    green=(g>=value)&(g>r*1.5)&(g>b*.8)
    features[f'strict_{value}']=[int(red.sum()),int(green.sum()),int(roi.shape[0]*roi.shape[1])]
 return features

def image_quality(frame):
 """Reject fully occluded/blank frames; this is not an obstacle classifier."""
 gray=cv2.cvtColor(cv2.resize(frame,(64,64)),cv2.COLOR_BGR2GRAY)
 spread=float(gray.std());p99=float(np.percentile(gray,99))
 return dict(valid=bool(spread>=3. and p99>=16.),gray_std=spread,p99=p99)
