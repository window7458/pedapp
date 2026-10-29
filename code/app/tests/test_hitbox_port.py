"""실시간으로 옮긴 v5_29 가 원본(오프라인 v5_29)과 같은 판정을 내는지 확인한다. 모델 · GPU · 영상 없이 돈다.

같은 가짜 입력(검출 · 광류 · 신호등 · 바닥 분할)을 ① 원본 run() 에 파일로 넣고, ② 실시간 코어에 프레임 하나씩 넣어서
  1) 경고 사건(묶기 전)이 한 글자 · 한 숫자까지 같은지
  2) 한 사건씩 처리하는 말하기 묶기(OnlineSpeech)가 원본 group_speech 와 같은지 (무작위 사건열 300개 포함)
를 본다.   python tests/test_hitbox_port.py
"""
from __future__ import annotations

import copy
import json
import math
import random
import sys
import tempfile
from pathlib import Path

import numpy as np

APP = Path(__file__).resolve().parents[1]
CODE = APP.parent
for p in (APP, CODE, CODE / "p0", CODE / "runtime"):
    sys.path.insert(0, str(p))

import hitbox_core as core  # noqa: E402
import hitbox_original_v5_29 as orig  # noqa: E402
from hitbox_trigger import OnlineSpeech, compose, prio_of  # noqa: E402

W, H, FPS, N = 540, 960, 15.0, 240
F = core.F_EQ_MM / 36.0 * max(W, H)
NAMES = {0: "person", 1: "car", 2: "bus", 3: "bicycle", 4: "obstacle", 5: "stairs"}
CID = {v: k for k, v in NAMES.items()}


def make_scene(seed=7):
    rng = np.random.default_rng(seed)
    yh = H / 2
    objs = [  # (tid, cls, H_m, W_m, t_on, t_off, X(t), Z(t))
        (1, "car", 1.5, 1.8, 0.0, 3.4, lambda t: 1.0, lambda t: 26 - 7.2 * t),
        (2, "obstacle", 1.4, 0.25, 0.0, 14.0, lambda t: -0.4, lambda t: 14 - 1.2 * t),
        (3, "person", 1.7, 0.5, 0.0, 16.0, lambda t: -2.5, lambda t: 9 - 0.2 * t),
        (4, "bus", 3.0, 2.5, 4.0, 8.5, lambda t: -6 + 3.0 * (t - 4.0), lambda t: 12.0),
        (5, "bicycle", 1.1, 0.6, 7.0, 10.5, lambda t: 0.3, lambda t: 11 - 4.2 * (t - 7.0)),
        (6, "obstacle", 1.2, 0.3, 2.0, 16.0, lambda t: 2.6, lambda t: 12 - 1.2 * (t - 2.0)),
        (7, "obstacle", 1.2, 0.3, 2.5, 16.0, lambda t: 2.7, lambda t: 15 - 1.2 * (t - 2.5)),
        (8, "person", 1.7, 0.5, 9.0, 13.0, lambda t: 0.1, lambda t: 10 - 3.2 * (t - 9.0)),
    ]
    frames, flows = [], {}
    pts = rng.uniform([20, 20], [W - 20, H - 20], size=(220, 2))
    foe = np.array([W / 2, H / 2])
    for fi in range(1, N + 1):
        t = round((fi - 1) / FPS, 4)
        dets = []
        for tid, cls, hm, wm, t0, t1, Xf, Zf in objs:
            if not t0 <= t <= t1:
                continue
            X, Z = Xf(t), Zf(t)
            if Z < 1.0:
                continue
            cx = W / 2 + F * X / Z
            y2 = yh + F * 1.7 / Z
            y1, hw = y2 - F * hm / Z, F * wm / Z / 2
            box = [round(cx - hw + rng.normal(0, 0.6), 1), round(y1 + rng.normal(0, 0.6), 1),
                   round(cx + hw + rng.normal(0, 0.6), 1), round(min(y2, H - 1) + rng.normal(0, 0.6), 1)]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            dets.append(dict(box=box, cls=CID[cls], conf=round(float(rng.uniform(0.35, 0.92)), 3), tid=tid))
        frames.append(dict(f=fi, t=t, dets=dets))
        if fi > 1:
            s = 1.0 + 0.0035 + rng.normal(0, 0.0004)
            shake = np.array([2.0 * math.sin(fi * 0.7), 1.5 * math.sin(fi * 0.9 + 1)])
            q = foe + (pts - foe) * s + shake + rng.normal(0, 0.25, pts.shape)
            flows[f"f{fi}"] = np.hstack([pts, q]).astype(np.float32)
    # 신호등: 3프레임마다 결과, 3 s 에 빨강 → 초록
    lights = []
    for fi in range(1, N + 1, 3):
        t = (fi - 1) / FPS
        st = "red" if t < 3.0 else "green"
        lights.append(dict(f=fi, t=round(t, 4), lights=[dict(box=[230.0, 300.0, 300.0, 390.0], conf=0.8, tid=11,
                                                              state=st, p=1.0)]))
    # 바닥: 5프레임마다. 아래쪽 가운데 인도, 왼쪽 아래 끝은 차도
    sfr, slab = [], []
    for fi in range(1, N + 1, 5):
        lab = np.zeros((240, 135), np.uint8)
        lab[130:, 20:120] = 1
        lab[130:, :20] = 2
        if fi > 120:
            lab[150:200, 20:120] = 3          # 한동안 앞쪽에 횡단보도
        sfr.append(fi)
        slab.append(lab)
    return frames, flows, lights, np.array(sfr), np.stack(slab)


def jn(x):
    return json.loads(json.dumps(x, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))


def key(r):
    return (r["t"], str(r.get("track_id")), r["cls"], r["scenario"], str(r.get("text")), str(r.get("state")))


def run_original(tmp: Path, frames, flows, lights, sfr, slab):
    meta = dict(video="synthetic.mp4", fps=FPS, n=N, w=W, h=H, names={str(k): v for k, v in NAMES.items()}, sec=1.0)
    (tmp / "dets.json").write_text(json.dumps(dict(meta=meta, frames=frames)), encoding="utf-8")
    np.savez_compressed(tmp / "flows.npz", **flows)
    (tmp / "events_B0.jsonl").write_text("", encoding="utf-8")
    (tmp / "lights_hsv.json").write_text(json.dumps(dict(stride=3, fps=FPS, frames=lights)), encoding="utf-8")
    np.savez_compressed(tmp / "surface2.npz", frames=sfr, labels=slab, stride=5)
    cap = []
    real = orig.group_speech
    orig.group_speech = lambda rt: (cap.append(copy.deepcopy(rt)), real(rt))[1]
    try:
        orig.run(str(tmp), make_video=False)
    finally:
        orig.group_speech = real
    return cap[0], json.loads((tmp / "events_runtime.json").read_text(encoding="utf-8"))["events"]


def run_streaming(frames, flows, lights, sfr, slab):
    c = core.V529Core(W, H)
    by_l = {x["f"]: x["lights"] for x in lights}
    rt_all = []
    for fr in frames:
        fi, t = fr["f"], fr["t"]
        dets = [dict(box=list(d["box"]), cls=NAMES[d["cls"]], conf=d["conf"], tid=d["tid"]) for d in fr["dets"]]
        k = int(np.searchsorted(sfr, fi, side="right")) - 1
        rt, _ = c.step_frame(t, fi, FPS, dets, flows.get(f"f{fi}"), by_l.get(fi), slab[k] if k >= 0 else None)
        rt_all += rt
    return rt_all


def test_core_equals_original():
    frames, flows, lights, sfr, slab = make_scene()
    with tempfile.TemporaryDirectory() as d:
        o_rt, o_grouped = run_original(Path(d), frames, flows, lights, sfr, slab)
    s_rt = run_streaming(frames, flows, lights, sfr, slab)
    a, b = sorted(jn(o_rt), key=key), sorted(jn(s_rt), key=key)
    print(f"[core] 원본 사건 {len(a)}개 / 실시간 {len(b)}개")
    assert len(a) > 0, "가짜 장면에서 사건이 하나도 안 나옴 — 시험이 의미 없음"
    assert a == b, "원본과 실시간 코어의 사건이 다름"
    kinds = sorted({(r["scenario"], r["cls"], r.get("motion"), r.get("action")) for r in a}, key=str)
    print("[core] 사건 종류:", kinds)
    return o_rt, o_grouped


def test_online_speech_equals_group_speech(o_rt, o_grouped):
    # 1) 가짜 장면의 사건열
    rt = sorted(copy.deepcopy(o_rt), key=lambda r: r["t"])
    sp = OnlineSpeech()
    got = []
    for r in rt:
        got += sp.feed(copy.deepcopy(r))
    got += sp.finish()
    a = sorted(jn(o_grouped), key=key)
    b = sorted(jn(got), key=key)
    assert a == b, f"묶기 결과가 다름\n원본 {a}\n실시간 {b}"
    print(f"[speech] 가짜 장면: 묶은 뒤 {len(a)}개 일치")
    # 2) 무작위 사건열 300개 — merge_rows · calm_veer · 묶기 · 급한 것 모두 건드림
    rng = random.Random(3)
    clocks = [0, 1, 2, 3, 9, 10, 11, 12]
    classes = ["car", "bicycle", "person", "obstacle", "pole_long", "pole_short", "stairs", "bus"]
    motions = ["static", "approaching", "crossing", "receding", "slow", "standing"]
    actions = ["none", "keep", "stop", "veer_left", "veer_right"]
    n_ev = 0
    for k in range(300):
        t, lst = 0.0, []
        for i in range(rng.randint(3, 40)):
            t += rng.choice([0.05, 0.1, 0.3, 0.6, 1.2, 3.0])
            sc = rng.choice(["S1", "S2", "S2", "S2", "S3", "S4", "S5"])
            r = dict(t=round(t, 3), cls=rng.choice(classes), scenario=sc, track_id=rng.randint(1, 8),
                     level=rng.choice(["warn", "danger"]), box=[1, 2, 3, 4], motion=rng.choice(motions),
                     action=rng.choice(actions), clock=rng.choice(clocks), dist=round(rng.uniform(1, 15), 1),
                     lat=rng.choice([None, round(rng.uniform(-3, 3), 2)]), row_n=rng.choice([0, 0, 1, 2, 3]))
            if sc == "S3":
                r.update(cls="traffic_light", state=rng.choice(["GO", "STOP"]), level="signal")
            if sc == "S5":
                r.update(cls="crosswalk", text="횡단보도 앞이에요.", urgent=True, track_id=None)
            lst.append(r)
        exp = orig.group_speech(copy.deepcopy(lst))
        sp = OnlineSpeech()
        got = []
        for r in lst:
            got += sp.feed(copy.deepcopy(r))
        got += sp.finish()
        assert sorted(jn(exp), key=key) == sorted(jn(got), key=key), f"무작위 {k}번 다름"
        n_ev += len(exp)
    print(f"[speech] 무작위 사건열 300개 일치 (묶인 사건 합계 {n_ev})")


def test_sentences(o_grouped):
    print("[문장] 예시:")
    seen = set()
    for r in o_grouped:
        s = compose(r)
        if s not in seen:
            seen.add(s)
            print(f"   p{prio_of(s)}  {s}")
    assert compose(dict(scenario="S2", cls="car", clock=12, motion="approaching", action="stop")) == "정면 자동차가 다가와요. 멈추세요."
    assert compose(dict(scenario="S2", cls="obstacle", clock=0, motion="static", action="veer_left")) == "경로상에 장애물이 있어요. 왼쪽으로 피하세요."
    assert compose(dict(scenario="S2", cls="pole_long", clock=1, motion="static", items=[
        dict(clock=1, cls="pole_long", motion="static"), dict(clock=11, cls="bicycle", motion="static")])
    ) == "1시 방향 긴 기둥, 11시 방향 자전거가 있어요."
    assert prio_of("정면 자동차가 다가와요. 멈추세요.") == 3 and prio_of("왼쪽으로 피하세요.") == 2 and prio_of("1시 방향 기둥이 있어요.") == 1


if __name__ == "__main__":
    o_rt, o_grouped = test_core_equals_original()
    test_online_speech_equals_group_speech(o_rt, o_grouped)
    test_sentences(o_grouped)
    print("\n전부 통과")
