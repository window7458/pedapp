"""JSON 4칸 → 한국어 문장 (모든 비교군 공통), 즉시 경고 템플릿, VLM 없는 기준선 PT (docs/09 §6-1)."""
from __future__ import annotations

KO_TARGET = {
    "person": "사람", "bicycle": "자전거", "scooter": "킥보드", "motorcycle": "오토바이", "car": "자동차",
    "bus": "버스", "other_vehicle": "차량", "tree": "나무", "pole": "기둥", "bollard": "볼라드", "sign": "표지판",
    "stall": "노점", "parked_bicycle": "세워진 자전거", "parked_scooter": "세워진 킥보드", "bench": "벤치",
    "barrier": "펜스", "stairs": "계단", "traffic_light": "신호등", "other": "장애물", "obstacle": "장애물",
}
KO_DIR = {"left": "왼쪽", "front": "정면", "right": "오른쪽"}
KO_MOTION = {
    "approaching": "다가오고 있어요", "static": "있어요", "receding": "멀어지고 있어요", "crossing": "앞을 지나가고 있어요",
    "red": "빨간불이에요", "green": "초록불이에요", "to_green": "초록불로 바뀌었어요", "to_red": "빨간불로 바뀌었어요",
    "up": "올라가는 계단이 있어요", "down": "내려가는 계단이 있어요",
}
KO_ACTION = {
    "stop": "멈추세요.", "veer_left": "왼쪽으로 피하세요.", "veer_right": "오른쪽으로 피하세요.",
    "wait": "기다리세요.", "cross": "건너가세요.", "caution": "조심해서 가세요.",
}

# 규칙표 (docs/09 §3-3) — PT의 행동, 채점의 허용 행동 집합에 같이 쓴다
def action_ok(scenario: str, direction: str, motion: str) -> list[str]:
    if scenario == "S1":
        if motion == "crossing":
            return ["stop", "wait"]
        return {"front": ["stop", "veer_left", "veer_right"], "left": ["stop", "veer_right"],
                "right": ["stop", "veer_left"]}[direction]
    if scenario == "S2":
        return {"front": ["veer_left", "veer_right", "stop"], "left": ["veer_right", "caution"],
                "right": ["veer_left", "caution"]}[direction]
    if scenario == "S3":
        return ["wait"] if motion in ("red", "to_red") else ["cross"]
    if scenario == "S4":
        return ["stop", "caution"]
    return []


def direction_of(box, W: float) -> str:
    """bbox 중심 x가 화면 폭의 1/3 미만 left, 2/3 초과 right (docs/09 §3-2)."""
    cx = (box[0] + box[2]) / 2 / W
    return "left" if cx < 1 / 3 else "right" if cx > 2 / 3 else "front"


def _josa(word: str) -> str:
    """받침 있으면 '이', 없으면 '가'."""
    c = word[-1]
    return "이" if "가" <= c <= "힣" and (ord(c) - 0xAC00) % 28 else "가"


def sentence(out: dict) -> str:
    tgt = KO_TARGET.get(out["target"], "장애물")
    d, m = KO_DIR[out["direction"]], out["motion"]
    if m in ("red", "green", "to_green", "to_red"):
        head = f"{d} 신호등이 {KO_MOTION[m]}."
    elif m in ("up", "down"):
        head = f"{d}에 {KO_MOTION[m]}."
    else:
        head = f"{d}에 {tgt}{_josa(tgt)} {KO_MOTION[m]}."
    return f"{head} {KO_ACTION[out['action']]}"


def warn_text(cls: str, direction: str) -> str:
    """즉시 경고 (VLM을 기다리지 않음). 클래스 × 방향 조합이라 wav를 미리 합성해 둘 수 있다."""
    return f"{KO_DIR[direction]} {KO_TARGET.get(cls, '장애물')}"


def scenario_of(cls: str) -> str:
    if cls in ("bicycle", "scooter", "motorcycle", "car", "bus", "other_vehicle", "person"):
        return "S1"
    return {"traffic_light": "S3", "stairs": "S4"}.get(cls, "S2")


def pt_output(event: dict, box, W: float) -> dict:
    """PT: VLM 없이 트리거 정보만으로 만든 같은 형식의 답 (VLM의 이득을 재기 위한 기준선)."""
    cls = event["cls"]
    d = direction_of(box, W) if box is not None else "front"
    sc = scenario_of(cls)
    motion = {"S3": "to_green" if event.get("state") == "GO" else "to_red", "S4": "down"}.get(sc, "approaching")
    return {"hazard": True, "target": "other" if cls == "obstacle" else cls, "direction": d,
            "motion": motion, "action": action_ok(sc, d, motion)[0]}
