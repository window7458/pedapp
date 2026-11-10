"""Risk wording uses the causal mobility state, without changing trigger gates."""
LABELS={0:'사람',1:'자전거',2:'킥보드',3:'오토바이',4:'차량',5:'버스',6:'차량',7:'장애물',8:'장애물'}

def describe_risk(event):
    if event['event'].startswith('light_'):
        return
    cid=event['class_id'];label=LABELS.get(cid,'장애물');state=event.get('motion_state','unknown')
    if cid in (1,2,3,4,5,6):
        if state=='parked':
            label=('세워진 ' if cid in (1,2,3) else '멈춰 있는 ')+label
            event['risk_type']='stationary_obstruction'
        elif state=='moving':
            label='이동 중인 '+label
            event['risk_type']='moving_vehicle'
        else:
            event['risk_type']='vehicle_motion_unresolved'
        event['message']=f"{event['clock']}시 방향, {label}에 유의하세요."
    else:
        event['risk_type']='person' if cid==0 else 'obstruction'
