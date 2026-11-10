"""보호자 푸시 알림 (Web Push). 보호자 화면을 닫아도 폰 알림으로 온다.

  · 키: 처음 한 번 VAPID 키를 만들어 certs/vapid.json 에 저장 (이 파일은 공유하지 말 것)
  · 구독: 보호자 화면에서 '푸시 알림 켜기' → 브라우저가 준 구독 정보를 alerts/push_subs.json 에 저장
  · 보내기: 새 알림이 생기면 구독마다 고른 단계(긴급만 / 긴급+주의 / 전체)에 맞을 때만 보낸다
  · 조건: HTTPS 주소여야 한다 (Cloudflare 터널 주소는 OK). 아이폰은 사파리에서 '홈 화면에 추가'한 앱에서만 된다 (iOS 16.4+)
"""
from __future__ import annotations

import base64
import json
import threading
from pathlib import Path

LV = {"urgent": "긴급", "caution": "주의", "info": "안내", "sos": "호출", "offline": "끊김", "online": "출발", "battery": "배터리", "notice": "소식"}
STRONG = {"urgent", "sos", "offline"}


def _b64u(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


class Push:
    def __init__(self, cert_dir: Path, data_dir: Path, subject: str = "mailto:sayeye@example.com", log=print):
        self.log, self.subject = log, subject
        self.key_file, self.sub_file = Path(cert_dir) / "vapid.json", Path(data_dir) / "push_subs.json"
        self.subs: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.ok = False
        try:
            from py_vapid import Vapid01
            from cryptography.hazmat.primitives import serialization
            if self.key_file.exists():
                k = json.loads(self.key_file.read_text())
                self.private_pem, self.public = k["private_pem"], k["public"]
            else:
                v = Vapid01()
                v.generate_keys()
                self.private_pem = v.private_pem().decode()
                raw = v.public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
                self.public = _b64u(raw)
                self.key_file.parent.mkdir(parents=True, exist_ok=True)
                self.key_file.write_text(json.dumps(dict(private_pem=self.private_pem, public=self.public)))
            self.vapid = Vapid01.from_pem(self.private_pem.encode())
            self.ok = True
        except Exception as e:                       # pywebpush 미설치 등 — 푸시만 빠진 채 서버는 돈다
            log(f"[push] 푸시 알림을 쓸 수 없어요: {e!r}  (pip install pywebpush)")
        if self.sub_file.exists():
            try:
                self.subs = json.loads(self.sub_file.read_text(encoding="utf-8"))
            except Exception:
                self.subs = {}

    def _save(self):
        self.sub_file.parent.mkdir(parents=True, exist_ok=True)
        self.sub_file.write_text(json.dumps(self.subs, ensure_ascii=False), encoding="utf-8")

    def subscribe(self, sub: dict, sens: str = "caution"):
        ep = sub.get("endpoint")
        if not ep:
            raise ValueError("구독 정보가 없어요")
        with self.lock:
            self.subs[ep] = dict(sub=sub, sens=sens)
            self._save()

    def unsubscribe(self, endpoint: str):
        with self.lock:
            self.subs.pop(endpoint, None)
            self._save()

    @staticmethod
    def wanted(level: str, sens: str) -> bool:
        if level in ("online", "notice"):        # 보행 시작 · 안부 알림은 '긴급만'이 아니면 보낸다
            return sens != "urgent"
        if sens == "urgent":
            return level in STRONG
        if sens == "caution":
            return level in STRONG or level in ("caution", "battery")
        return True

    def on_alert(self, a: dict, merged: bool):
        if not self.ok or not self.subs:
            return
        if merged and a.get("count", 1) > 2 and a["count"] % 3:          # 이어지는 같은 알림은 몇 번에 한 번만
            return
        if a.get("clip_only"):
            return
        body = dict(title=f"SayEye · {LV.get(a['level'], '')}", body=a["text"], tag=f"sayeye-{a['level']}",
                    level=a["level"], id=a["id"], url="/guardian")
        threading.Thread(target=self._send_all, args=(a["level"], body), daemon=True).start()

    def _send_all(self, level: str, body: dict):
        from pywebpush import WebPushException, webpush
        with self.lock:
            targets = [(ep, s) for ep, s in self.subs.items() if self.wanted(level, s.get("sens", "caution"))]
        dead = []
        for ep, s in targets:
            try:
                webpush(s["sub"], json.dumps(body, ensure_ascii=False), vapid_private_key=self.vapid,
                        vapid_claims={"sub": self.subject}, ttl=120,
                        headers={"Urgency": "high" if level in STRONG else "normal"})
            except WebPushException as e:
                code = getattr(getattr(e, "response", None), "status_code", None)
                if code in (404, 410):               # 구독이 만료됨 → 지움
                    dead.append(ep)
                else:
                    self.log(f"[push] 보내기 실패 {code}: {e}")
            except Exception as e:
                self.log(f"[push] 보내기 실패: {e!r}")
        for ep in dead:
            self.unsubscribe(ep)
