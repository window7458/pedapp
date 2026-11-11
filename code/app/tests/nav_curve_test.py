import sys, math, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nav import _bends, Nav
lat0, lon0 = 37.45, 126.65
k, kx = 111320.0, 111320.0*math.cos(math.radians(lat0))
P = lambda x,y: [lat0+y/k, lon0+x/kx]
# 북쪽 100m 직진 → 반지름 40m 로 오른쪽 90° 휘는 길 → 동쪽 100m
line = [P(0,y) for y in range(0,101,10)]
line += [P(40-40*math.cos(math.radians(a)), 100+40*math.sin(math.radians(a))) for a in range(10,91,10)]
line += [P(40+x,140) for x in range(10,101,10)]
end = dict(lat=line[-1][0], lon=line[-1][1], kind="arrive", text="도착했어요.")
s = _bends(line, [end])
for st in s: print(st["kind"], st["text"])
assert [x["kind"] for x in s] == ["curve","arrive"] and "오른쪽" in s[0]["text"]
# 이미 TMAP 회전 안내가 그 자리에 있으면 커브 안내를 안 넣음
turn = dict(lat=P(12,128)[0], lon=P(12,128)[1], kind="turn", text="오른쪽")
assert [x["kind"] for x in _bends(line, [turn, end])] == ["turn","arrive"]
# 직선 길이면 아무것도 안 넣음
st = [P(0,y) for y in range(0,301,10)]
assert len(_bends(st, [end])) == 1
print("synthetic ok")
