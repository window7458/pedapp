"""Perspective path corridor followed by causal bbox confirmation.

The ground/path estimator is unchanged. No per-object metric distance, TTC,
extra detector, class correction, source name, or GT enters this rule.
"""
from dataclasses import dataclass,field
import math,numpy as np
from bbox_evidence import Evidence
from entry_rule import EntryRule

@dataclass(frozen=True)
class Config:
    roi: dict = field(default_factory=lambda:dict(near_half=.20,far_half=.06,top=.5,gain=.2))
    groups: dict = field(default_factory=lambda:dict(
        obstacle=dict(height=.3,width=.5,growth=.5,speed=.15,confirm=.3,aspect=1.),
        vehicle=dict(height=.2,width=2.,growth=.5,speed=.3,confirm=0.,aspect=0.),
        close_obstacle=dict(height_mode='frame',height=.6,width=.35,growth=.05,speed=.05,confirm=.1,aspect=0.)))
    rearm_area_ratio: float = 0.0
    same_class_cooldown_s: float = 1.5
    urgent: dict = field(default_factory=lambda:dict(enabled=True,bottom=.9,height=.3,confirm=.1,inward_speed=.3))
    crossing: dict = field(default_factory=lambda:dict(enabled=True,height=.2,inward_speed=.2,confirm=.1))

def box_overlap(a,b):
    a=np.asarray(a);b=np.asarray(b)
    inter=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
    return float(inter/max(1.,min(np.prod(a[2:]-a[:2]),np.prod(b[2:]-b[:2]))))

def path_polygon(context,config,width,height):
    """Floor-like trapezoid, not a calibrated metric footprint.

    Near edge stays at the user's feet. The existing path moves the far edge;
    walking speed extends look-ahead. Rendering and membership share these edges.
    """
    p=config.roi;short=min(width,height)
    shift=float(np.clip(p['gain']*(context['camera_cx']-width/2+context['focal_px']*math.tan(context['path_heading_rad'])),-.08*width,.08*width))
    top=float(np.clip(p['top']-.04*(context['walk_mps']-1.2),.35,.7))*height
    near=p['near_half']*short;far=p['far_half']*short;cx=width/2
    return np.array([[cx-near,height],[cx+near,height],
                     [cx+shift+far,top],[cx+shift-far,top]],dtype=float)

def path_rectangle(context,config,width,height):
    """Bounding rectangle for diagnostics only; never used as the region gate."""
    p=path_polygon(context,config,width,height)
    return [float(p[:,0].min()),float(p[:,1].min()),float(p[:,0].max()),float(p[:,1].max())]

def clock_direction(box,width,focal_px):
    """Approximate user/camera-relative clock direction, frozen at alert time."""
    angle=math.degrees(math.atan2((box[0]+box[2])/2-width/2,max(1.,focal_px)))
    k=0 if abs(angle)<=5 else min(3,max(1,int((abs(angle)+15)//30)))
    return (12 if k==0 else k if angle>0 else 12-k)

def path_geometry(box,context,width,height,config):
    x1,y1,x2,y2=box;p=path_polygon(context,config,width,height)
    y=float(np.clip(y2,p[2,1],height));a=(y-p[2,1])/max(1.,height-p[2,1])
    left=float(p[3,0]*(1-a)+p[0,0]*a);right=float(p[2,0]*(1-a)+p[1,0]*a)
    # Compare the bbox's ground-contact span with the corridor at that depth.
    # A wide car covering the path must not fail merely because its bbox is wide.
    overlap=max(0.,min(x2,right)-max(x1,left))/max(1.,min(x2-x1,right-left))
    inside=overlap>=.25 and y2>=p[2,1] and y1<height
    short=min(width,height);side=((x1+x2)/2-(left+right)/2)/short
    return dict(in_path=bool(inside),overlap=overlap,polygon=p.tolist(),footprint=[left,y,right,y],rectangle=path_rectangle(context,config,width,height),side=side,
                near=bool(inside and y2>=config.urgent['bottom']*height),z=None,zone_m=None)

class SimpleTrigger:
    def __init__(self,config=None):
        self.config=config or Config();self.evidence=Evidence();self.entry=EntryRule(self.config.crossing);self.states={};self.recent=[];self.last_class={};self.suppressed=[]
    def step(self,frame,context):
        rows=self.evidence.step(frame,context);t=frame['time_s'];w,h=frame['width'],frame['height']
        mat=np.asarray(frame['matrix']) if frame.get('matrix') is not None and not context.get('camera_shock') else np.eye(2,3)
        self.recent=[e for e in self.recent if t-e['time_s']<1.5]
        for e in self.recent:
            e['corners']=e['corners']@mat[:,:2].T+mat[:,2]
            e['box']=np.r_[e['corners'].min(0),e['corners'].max(0)].tolist()
        for tid,state in list(self.states.items()):
            if t-state['last']>2:del self.states[tid]
        events=[];decisions=[]
        for r in rows:
            tid=r['track_id'];state=self.states.setdefault(tid,dict(birth=t,last=t,since={},fired=False,inside=[],peak_area=0.))
            if r['gap']>.4 or context.get('camera_shock',False):state['since'].clear();state['inside'].clear()
            state['last']=t;geo=path_geometry(r['box'],context,w,h,self.config)
            if 'born_near' not in state:
                state['born_near']=geo['near'] and r['bh']>=self.config.urgent['height']
            state['inside']=[o for o in state['inside'] if t-o[0]<=.65]
            had_outside=any(not inside for _,inside in state['inside'])
            state['inside'].append((t,geo['in_path']))
            area=(r['box'][2]-r['box'][0])*(r['box'][3]-r['box'][1])
            if state['fired']:
                state['peak_area']=max(state['peak_area'],area)
                if self.config.rearm_area_ratio>0 and area<self.config.rearm_area_ratio*state['peak_area']:
                    state['fired']=False;state['since'].clear()
            observed=r['observed'] and r['n']>=3 and r['span']>=.19 and r['confidence']>=.5
            ready=[]
            group_names=['obstacle','close_obstacle'] if r['class_id'] in (7,8) else ['vehicle']
            for name in group_names:
                p=self.config.groups[name]
                height_threshold=p['height']*(h/r['short'] if p.get('height_mode')=='frame' else 1.)
                growth=r['growth']*(r['bh'] if p.get('growth_mode')=='absolute' else 1.)
                good=observed and geo['in_path'] and r['bh']>=height_threshold and r['bw']<=p['width'] and r['aspect']>=p['aspect'] and growth>=p['growth'] and r['speed']<=p['speed']
                if not good:state['since'].pop(name,None)
                else:
                    state['since'].setdefault(name,t)
                    if t-state['since'][name]>=p['confirm']-1e-8:ready.append(name)
            u=self.config.urgent
            inward=-np.sign(geo['side'])*r['vx']
            entry_evidence=self.entry.step(r,geo)
            # Urgent branch has the same hard region gate. It cannot warn outside.
            # A detection gap is not a new object's first appearance.
            # Stable or growing close boxes qualify; clearly shrinking ones do not.
            fresh=state['born_near'] and t-state['birth']<=.8 and r['growth']>=-.1
            entry=had_outside and inward>=u['inward_speed']
            urgent=u['enabled'] and observed and geo['in_path'] and geo['near'] and r['bh']>=u['height'] and r['camera_ok'] and (fresh or entry)
            if not urgent:state['since'].pop('urgent',None)
            else:
                state['since'].setdefault('urgent',t)
                if t-state['since']['urgent']>=u['confirm']-1e-8:ready.insert(0,'urgent')
            if entry_evidence:ready.insert(0,'sudden_entry')
            reason=ready[0] if ready else 'observing' if geo['in_path'] else 'outside_path'
            decisions.append(dict(track_id=tid,**{'class':r['class']},box=r['box'],reason=reason,growth=r['growth'],residual_speed=r['speed'],**geo))
            # One common final gate protects every branch, including future additions.
            if not geo['in_path'] or not ready or (state['fired'] and not entry_evidence):continue
            state['fired']=True;state['peak_area']=area
            if any((e['class']==r['class'] or reason=='sudden_entry' or e['rule_group']=='sudden_entry') and box_overlap(e['box'],r['box'])>=.6 for e in self.recent):continue
            if reason!='sudden_entry' and t-self.last_class.get(r['class'],-math.inf)<self.config.same_class_cooldown_s:
                self.suppressed.append(dict(frame=r['frame'],time_s=t,track_id=tid,**{'class':r['class']},reason='same_class_cooldown',box=r['box']))
                continue
            self.last_class[r['class']]=t
            keys=['box','class_id','class','confidence','track_id','order','frame','time_s']
            event={k:r[k] for k in keys};event.update(event='approach',rule_group=reason,growth_per_second=r['growth'],center_speed=r['speed'],roi_box=geo['rectangle'],roi_polygon=geo['polygon'],roi_footprint=geo['footprint'],roi_overlap=geo['overlap'],in_path=geo['in_path'],clock=clock_direction(r['box'],w,context['focal_px']),priority='urgent' if reason in ('sudden_entry','urgent') else 'normal',walk_mps=context['walk_mps'],path_valid=context.get('path_valid',False))
            if entry_evidence:event.update(entry_evidence=entry_evidence,notice_category='object')
            events.append(event);x1,y1,x2,y2=r['box']
            self.recent.append(dict(event,corners=np.array([[x1,y1],[x2,y1],[x1,y2],[x2,y2]])))
        return events,decisions
