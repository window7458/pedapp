"""hitbox_original_v5_29.py (오프라인 v5_29) → hitbox_core.py (프레임 하나씩 들어오는 실시간 판정 코어) 자동 변환.

손으로 다시 쓰면 판정이 조금씩 달라진다. 그래서 v5_29의 상수 · 함수 · `for fr in frames:` 루프 본문 · 이벤트 만들기를
줄 그대로 가져오고, 아래 '입력을 공급하는 곳'만 바꿔 끼운다. 바꾼 곳은 REPL 표(PATCHES)에 전부 있고,
원본에서 그 줄을 찾지 못하면 (= 원본이 바뀜) 에러로 멈춘다.

    python hitbox_build.py            # hitbox_core.py 를 다시 만든다
"""
from __future__ import annotations

import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "hitbox_original_v5_29.py"
DST = HERE / "hitbox_core.py"

# (원본 조각, 바꿀 조각). 루프 본문 안에서 정확히 count 번 나와야 한다.
PATCHES = [
    # ── 프레임 입력: 파일(dets.json)에서 읽던 것을 인자로 ─────────────────────────────
    ('        t, fi = fr["t"], fr["f"]\n', '', 1),
    # ── 신호등: 미리 저장한 lights_hsv.json 대신, 새 결과가 왔을 때만 갱신 ─────────────
    ('        if fi in lights_by_f:\n            cur_lights = lights_by_f[fi]\n',
     '        if new_lights is not None:\n            cur_lights = new_lights\n            self.light_t = t\n', 1),
    ('        elif fi - max([k for k in (fi - d for d in range(stride)) if k in lights_by_f] or [-99]) >= stride:\n',
     '        elif t - self.light_t > LIGHT_STALE_S:\n', 1),
    # ── 광류 쌍: flows.npz 대신 인자로 ───────────────────────────────────────────────
    ('        pairs = flows[f"f{fi}"] if f"f{fi}" in flows else None\n', '', 1),
    # ── 바닥 분할: surface2.npz 에서 프레임 번호로 찾던 것을 가장 최근 결과로 ──────────────
    ('        if surf_f is not None:\n            k_ = int(np.searchsorted(surf_f, fi, side="right")) - 1\n',
     '        if surf_cur is not None:\n            k_ = 0\n            surf_l = [surf_cur]\n', 1),
    # ── 검출: 클래스 번호 → 이름 (실시간 추적기는 이름을 바로 준다) ──────────────────────
    ('names[d["cls"]]', 'd["cls"]', 2),
    ('sorted(fr["dets"], key=lambda d: -d.get("conf", 1.0))', 'sorted(dets, key=lambda d: -d.get("conf", 1.0))', 1),
]

# 이벤트 만들기(원본 126–172행)에서 바꾸는 곳: 영상이 끝난 뒤 한꺼번에 → 이 프레임 것만 바로
RT_PATCHES = [
    ('per_frame_draw[e["frame"] - 1]', 'per_frame_draw[-1]', 3),
]

# 루프 앞에서 run() 이 준비하던 상태 → self 속성
STATE_INIT = '''
        self.W, self.H = W, H
        self.f_px = F_EQ_MM / 36.0 * max(W, H)
        self.tracks = {}
        self.foe = np.array([W / 2, H / 2], float)
        self.walk = WALK_DEFAULT
        self.veh_near_t, self.xwalk_t, self.xwalk_tid = -1e9, -1e9, None
        self.shake_cum, self.shake_base, self.y_hor_prev = np.zeros(2), np.zeros(2), H / 2
        self.light_st, self.cur_lights, self.light_t = defaultdict(LightState), [], -1e9
        self.n_offroad = self.n_offpath = 0
        self.xw_state, self.xw_said = None, {}
        self.walk_edge = [None, None]
        self.crowd_hist, self.crowd_said = deque(maxlen=90), -1e9
        self.last_walk_side, self.road_t0 = -1e9, 0.0
        self.turning, self.heading, self.prev_t = False, 0.0, None
        self.roads, self.timing, self.walks = deque(maxlen=600), deque(maxlen=600), deque(maxlen=600)
        self.cam_h_obs = deque(maxlen=600)
        self.per_frame_draw = deque(maxlen=8)
'''

# step_frame 앞에서 self → 지역 변수로 꺼내고, 끝에서 다시 self 로 돌려놓을 이름
LOCALS = ["W", "H", "f_px", "tracks", "foe", "walk", "veh_near_t", "xwalk_t", "xwalk_tid", "shake_cum", "shake_base",
          "y_hor_prev", "light_st", "cur_lights", "n_offroad", "n_offpath", "xw_state", "xw_said", "walk_edge",
          "crowd_hist", "crowd_said", "last_walk_side", "road_t0", "turning", "heading", "prev_t", "roads", "timing",
          "walks", "cam_h_obs", "per_frame_draw"]
STORE_BACK = [n for n in LOCALS if n not in ("W", "H", "f_px")]


def indent(text: str, n: int) -> str:
    pad = " " * n
    return "".join((pad + ln if ln.strip() else ln) for ln in text.splitlines(keepends=True))


def apply(text: str, patches, where: str) -> str:
    for old, new, count in patches:
        got = text.count(old)
        if got != count:
            raise SystemExit(f"[{where}] 원본에서 바꿀 곳을 못 찾음 (기대 {count}개, 실제 {got}개):\n{old!r}")
        text = text.replace(old, new)
    return text


def build() -> str:
    lines = SRC.read_text(encoding="utf-8-sig").splitlines(keepends=True)

    def find(pat, start=0):
        for i in range(start, len(lines)):
            if re.match(pat, lines[i]):
                return i
        raise SystemExit(f"원본에서 '{pat}' 를 찾지 못함")

    i_run = find(r"def run\(")
    i_for = find(r"    for fr in frames:", i_run)
    i_walks = find(r"        walks\.append\(walk\)", i_for)
    i_rt = find(r"    rt = \[\]", i_walks)
    i_group = find(r"    pe_ = sorted", i_rt)

    head = "".join(lines[:i_run])                         # 상수 · 함수 · 클래스 (run 앞까지 그대로)
    body = "".join(lines[i_for + 1:i_walks + 1])          # 루프 본문 (8칸 들여쓰기)
    rtb = "".join(lines[i_rt:i_group])                    # 이벤트 만들기 (4칸 들여쓰기)

    body = apply(body, PATCHES, "loop")
    rtb = apply(rtb, RT_PATCHES, "rt")
    rtb = indent(rtb, 4)                                  # 4칸 → 8칸 (메서드 안)

    prelude = "".join(f"        {n} = self.{n}\n" for n in LOCALS)
    prelude += "        evA, evB, evL, evX = [], [], [], []      # 이 프레임에서 새로 난 것만\n"
    store = "".join(f"        self.{n} = {n}\n" for n in STORE_BACK)
    prune = ('        for k_ in [k for k, v in tracks.items() if t - v.last_t > TRACK_KEEP_S]:   # 오래 안 보인 트랙 정리\n'
             '            del tracks[k_]\n'
             '        if len(light_st) > 64:\n'
             '            live_ = {L["tid"] for L in cur_lights}\n'
             '            for k_ in [k for k in light_st if k not in live_]:\n'
             '                del light_st[k_]\n')

    out = [
        '"""AUTO-GENERATED by hitbox_build.py — 직접 고치지 마세요 (고치려면 원본이나 hitbox_build.py 를 고치고 다시 생성).\n\n',
        'v5_29 히트박스 트리거의 실시간 코어: 프레임 하나가 들어올 때마다 step_frame() 이 판정 한 번을 돌린다.\n',
        '판정 코드(상수 · 함수 · 루프 본문 · 이벤트 만들기)는 hitbox_original_v5_29.py 와 줄 단위로 같다.\n',
        '"""\n',
        head.split('"""', 2)[2].lstrip("\n") if head.lstrip().startswith('"""') else head,
        "\n\n# ---------------- 실시간 코어 (hitbox_build.py 가 붙임) ----------------\n",
        "LIGHT_STALE_S = 1.2     # 신호등 결과가 이만큼 오래 안 오면 '안 보임' (원본: 저장 간격 STRIDE 프레임)\n",
        "TRACK_KEEP_S = 8.0      # 이만큼 안 보인 트랙은 버림 (원본은 영상 끝까지 보관 — 실시간은 메모리가 늘면 안 됨)\n\n\n",
        "class V529Core:\n",
        '    """상태를 속성으로 들고, step_frame() 이 원본 루프 한 바퀴를 돈다."""\n\n',
        "    def __init__(self, W, H):\n",
        STATE_INIT.lstrip("\n"),
        "\n    def step_frame(self, t, fi, fps, dets, pairs, new_lights, surf_cur):\n",
        '        """t 영상 시각(s) · fi 프레임 번호 · dets [{box, cls(이름), conf, tid}] · pairs 배경 광류 쌍(N×4) 또는 None ·\n',
        '        new_lights 새 신호등 결과(없으면 None) · surf_cur 바닥 분할 라벨(없으면 None).\n',
        '        반환: (rt, pf) — rt = 이 프레임에서 난 경고 사건 목록(묶기 전), pf = 이 프레임의 그림 정보."""\n',
        prelude,
        body,
        prune,
        "        rt = []\n",
        rtb.replace("        rt = []\n", "", 1),
        store,
        "        return rt, per_frame_draw[-1]\n",
    ]
    return "".join(out)


if __name__ == "__main__":
    DST.write_text(build(), encoding="utf-8")
    print("wrote", DST, DST.stat().st_size, "bytes")
