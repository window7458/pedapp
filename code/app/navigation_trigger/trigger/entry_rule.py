"""Class-independent observed entry into the walking corridor.

No expansion or metric-distance requirement. A small outside history and
camera-compensated inward motion distinguish entry from border jitter/panning.
"""
class EntryRule:
    def __init__(self,config):
        self.config=config;self.states={}

    def step(self,row,geometry):
        t=row['time_s'];tid=row['track_id'];p=self.config
        for key,s in list(self.states.items()):
            if t-s['last']>2:del self.states[key]
        s=self.states.setdefault(tid,dict(last=t,outside=[],since=None))
        s['last']=t
        if row['gap']>.4 or not row['observed'] or not row['camera_ok']:
            s['outside'].clear();s['since']=None;return None
        s['outside']=[o for o in s['outside'] if t-o['time_s']<=.65]
        if not geometry['in_path']:
            edge='top' if row['box'][3]<geometry['polygon'][2][1] else 'left' if geometry['side']<0 else 'right'
            s['outside'].append(dict(time_s=t,edge=edge));s['since']=None
            return None
        outside=s['outside']
        ready=row['n']>=3 and row['span']>=.19 and row['confidence']>=.5
        if not p['enabled'] or not ready or row['bh']<p['height'] or len(outside)<2 or outside[-1]['time_s']-outside[0]['time_s']<.09:
            s['since']=None;return None
        # Keep the entry edge during confirmation, including when the centre
        # crosses the corridor centre. Vertical entry is supported as well.
        edge=outside[-1]['edge']
        inward={'left':row['vx'],'right':-row['vx'],'top':row['vy']}[edge]
        if inward<p['inward_speed']:
            s['since']=None;return None
        if s['since'] is None:s['since']=t
        if t-s['since']<p['confirm']-1e-8:return None
        evidence=dict(edge=edge,outside_start_s=outside[0]['time_s'],outside_end_s=outside[-1]['time_s'],inside_start_s=s['since'],inward_speed=inward)
        s['outside'].clear();s['since']=None
        return evidence
