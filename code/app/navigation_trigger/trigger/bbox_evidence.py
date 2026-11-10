"""Causal bbox evidence; no label, video ID or future frame enters features."""
from collections import deque
import math,numpy as np
class Evidence:
 def __init__(self,history=.65,max_gap_s=.4,stale_s=2.):
  self.states={};self.history=history;self.max_gap_s=max_gap_s;self.stale_s=stale_s;self.last=None;self.recovery=-1e9
 def step(self,f,ctx):
  t=f['time_s'];short=min(f['width'],f['height'])
  if self.last is not None and t<=self.last:raise ValueError('Nonmonotonic timestamp')
  gap=0 if self.last is None else t-self.last;self.last=t
  shock=ctx.get('camera_shock',False)
  if shock or gap>self.max_gap_s+1e-8:
   self.recovery=t+.35
   for s in self.states.values():s['q'].clear()
  m=np.asarray(f['matrix']) if f.get('matrix') is not None and not shock else np.eye(2,3)
  for tid,s in list(self.states.items()):
   if t-s['last']>self.stale_s:del self.states[tid];continue
   for o in s['q']:o['p']=m[:,:2]@o['p']+m[:,2]
  rows=[]
  for d in f['detections']:
   if d['class_id']==9 or d.get('track_id') is None:continue
   tid=d['track_id'];b=np.asarray(d['box'],float)
   if np.any(b[2:]<=b[:2]) or not np.isfinite(b).all():continue
   s=self.states.setdefault(tid,dict(first=t,last=t,q=deque()))
   gap=t-s['last'];s['last']=t
   if gap>self.max_gap_s+1e-8:s['q'].clear();s['first']=t
   p=(b[:2]+b[2:])/2;bw,bh=b[2:]-b[:2]
   q=s['q'];q.append(dict(t=t,p=p.copy(),raw=p.copy(),a=math.log(bw*bh),w=math.log(bw),h=math.log(bh),confidence=d['confidence']))
   while q and t-q[0]['t']>self.history+1e-8:q.popleft()
   span=t-q[0]['t'];growth=0.;height_growth=0.;raw_speed=speed=1e6;vx=vy=0.;aspect_change=0.
   if len(q)>=2 and span>=.09:
    tt=np.array([o['t'] for o in q]);tt-=tt.mean();den=tt@tt
    growth=float(tt@np.array([o['a'] for o in q])/den)
    height_growth=float(tt@np.array([o['h'] for o in q])/den)
    v=tt@np.array([o['p'] for o in q])/den/short;vr=tt@np.array([o['raw'] for o in q])/den/short
    vx=float(v[0]);vy=float(v[1]);speed=float(np.linalg.norm(v));raw_speed=float(np.linalg.norm(vr))
    aspect_change=float(abs(tt@np.array([o['w']-o['h'] for o in q])/den))
   rows.append(dict(d,frame=f['frame'],time_s=t,width=f['width'],height=f['height'],short=short,growth=growth,height_growth=height_growth,width_growth=growth-height_growth,median_confidence=float(np.median([o['confidence'] for o in q])),speed=speed,raw_speed=raw_speed,vx=vx,vy=vy,aspect_change=aspect_change,n=len(q),span=span,age=t-s['first'],observed=not shock and t>=self.recovery,gap=gap,bw=bw/short,bh=bh/short,aspect=bh/bw,walk=ctx['walk_mps'],heading=ctx['path_heading_rad'],focus=ctx['camera_cx'],focal=ctx['focal_px'],camera_ok=f['camera']['camera_ok']))
  return rows
