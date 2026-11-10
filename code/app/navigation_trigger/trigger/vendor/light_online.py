"""Streaming counterpart of lights.light_events; current/past frames only."""
from collections import defaultdict,deque

class LightRule:
 def __init__(self,mode='hsv_100_0.1',fraction=.002,floor=2,dominance=1.2,stable=2,minconf=.15,maxaspect=1.5,maxgap=.5,global_cooldown=2.,ignore_unknown=False,stable_red=None,max_known_gap=.3,max_state_gap=2.,max_confirmation_span=.9):
  self.max_state_gap=max_state_gap;self.max_confirmation_span=max_confirmation_span
  self.max_known_gap=max_known_gap;self.last_known={};self.mode=mode;self.fraction=fraction;self.floor=floor;self.dominance=dominance;self.stable=stable;self.stable_red=stable if stable_red is None else stable_red
  self.minconf=minconf;self.maxaspect=maxaspect;self.maxgap=maxgap;self.cooldown=global_cooldown;self.ignore_unknown=ignore_unknown
  self.queues=defaultdict(lambda:deque(maxlen=max(stable,self.stable_red)));self.times=defaultdict(lambda:deque(maxlen=max(stable,self.stable_red)));self.state={};self.last={};self.fired=set();self.global_last={}
 def step(self,f):
  events=[];self.diagnostics=[]
  for d in f['detections']:
   if d['class_id']!=9:continue
   tid=d['track_id'];t=f['time_s'];x1,y1,x2,y2=d['box']
   diagnostic=dict(track_id=tid,box=d['box'],source=d.get('source','custom'),time_s=t,previous=self.state.get(tid))
   self.diagnostics.append(diagnostic)
   if t-self.last_known.get(tid,t)>self.max_state_gap:
    self.state.pop(tid,None);self.queues[tid].clear();self.times[tid].clear()
    self.fired={key for key in self.fired if key[0]!=tid}
    diagnostic['state_reset']='color_history_expired'
   if d.get('peak_confidence',d['confidence'])<self.minconf:
    diagnostic['reason']='low_confidence';continue
   if (x2-x1)/max(1,y2-y1)>self.maxaspect:
    diagnostic['reason']='unsupported_light_shape';continue
   if not d.get('color_features'):
    diagnostic['reason']='missing_current_color';continue
   if self.mode=='adaptive_emission':
    low=d['color_features']['strict_100'];high=d['color_features']['strict_140']
    r=high[0] if low[2]>=1024 else low[0];g=low[1];area=low[2]
   else:r,g,area=d['color_features'][self.mode]
   mn=max(self.floor,int(area*self.fraction))
   color='red' if r>=mn and r>self.dominance*g else ('green' if g>=mn and g>self.dominance*r else 'unknown')
   if (tid in self.last and t-self.last[tid]>self.maxgap) or (self.ignore_unknown and t-self.last_known.get(tid,-1e20)>self.max_known_gap+1e-9):self.queues[tid].clear();self.times[tid].clear()
   self.last[tid]=t
   diagnostic.update(color=color,red_pixels=r,green_pixels=g,min_pixels=mn)
   if color=='unknown' and self.ignore_unknown:
    diagnostic['reason']='unknown_color';continue
   if color!='unknown':self.last_known[tid]=t
   q=self.queues[tid];q.append(color);self.times[tid].append(t)
   required=self.stable_red if color=='red' and self.state.get(tid)=='green' else self.stable
   diagnostic.update(required_observations=required,recent_colors=list(q))
   if len(q)<required or color=='unknown' or any(c!=color for c in list(q)[-required:]):
    diagnostic['reason']='unknown_color' if color=='unknown' else 'awaiting_stable_color';continue
   if t-list(self.times[tid])[-required]>self.max_confirmation_span+1e-8:
    diagnostic['reason']='sparse_color_observations';continue
   prev=self.state.get(tid);self.state[tid]=color
   if prev is None or prev==color:
    diagnostic['reason']='initial_color' if prev is None else 'unchanged_color';continue
   kind=f'light_{prev}_to_{color}';key=(tid,kind)
   # A later real signal cycle must be eligible again for the same track.
   # Stable-state changes and the time cooldown still suppress duplicates.
   self.fired.discard((tid,f'light_{color}_to_{prev}'))
   if key in self.fired:
    diagnostic['reason']='already_announced_transition';continue
   self.fired.add(key)
   if t-self.global_last.get(kind,-1e20)<self.cooldown:
    diagnostic['reason']='transition_cooldown';continue
   self.global_last[kind]=t
   diagnostic['reason']='warned'
   events.append(dict(d,event=kind,color=color,frame=f['frame'],time_s=t,confirmation_span_s=t-list(self.times[tid])[-required],confirmation_observations=required))
  if not self.diagnostics:self.diagnostics=[dict(time_s=f['time_s'],reason='no_light_detection')]
  return events
 def prune(self,t,ttl=30.):
  expired={tid for tid,last in self.last.items() if t-last>ttl}
  for tid in expired:
   self.last_known.pop(tid,None);self.last.pop(tid,None);self.queues.pop(tid,None);self.times.pop(tid,None);self.state.pop(tid,None)
  self.fired={key for key in self.fired if key[0] not in expired}
