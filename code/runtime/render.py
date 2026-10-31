"""시연 영상: run_demo 결과(dets · log · timeline) → demo.mp4 (원본 영상 + 박스 + 트리거 표시 + 자막 + 실제 재생된 소리).

소리는 timeline의 실제 재생 구간 그대로 (끊긴 설명은 끊긴 지점까지) 섞는다 → 영상에서 들리는 것 = 시스템이 낸 것.
    python render.py <clip> <run 폴더>       # run_demo.py --render 가 자동으로 부름
"""
from __future__ import annotations

import json
import subprocess
import sys
import wave
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import core  # noqa: F401
from common import OUT, clip_info, iter_frames

SR = 22050
_FONTS = ["C:/Windows/Fonts/malgunbd.ttf", "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
          "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"]
FONT = next((f for f in _FONTS if Path(f).exists()), _FONTS[0])      # 리눅스 서버: apt install fonts-nanum
PANEL_W = 680                                                      # 오른쪽 파이프라인 패널 폭 (px)


def read_wav(path):
    with wave.open(str(path)) as w:
        sr, n, ch, sw = w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()
        x = np.frombuffer(w.readframes(n), dtype={1: np.int8, 2: np.int16, 4: np.int32}[sw]).astype(np.float32)
    x = x.reshape(-1, ch).mean(1) / float(2 ** (8 * sw - 1))
    if sr != SR:
        from scipy.signal import resample_poly
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g)
    return x


def mix_audio(timeline, dur):
    out = np.zeros(int((dur + 1.0) * SR), np.float32)
    for it in timeline:
        if "t_start" not in it:
            continue
        x = read_wav(it["wav"])
        a = int(it["t_start"] * SR)
        n = min(len(x), int((it["t_end"] - it["t_start"]) * SR), len(out) - a)
        if n > 0:
            out[a:a + n] += x[:n]
    return np.clip(out, -1, 1)


def write_wav(path, x):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((x * 32767).astype(np.int16).tobytes())


def _fmt_out(o):
    if not o:
        return "형식 오류"
    h = "" if o.get("hazard") is None else f" 위험={o.get('hazard')}"         # 2칸 출력(h2)엔 위험 칸이 없음
    return f"대상={o.get('target')} 움직임={o.get('motion')}{h}"


def pipeline_lines(rows):
    """log.jsonl → (영상 시각, 단계, 글) — 패널에 단계별로 무엇이 들어가고 나오는지 보여줌."""
    out = []
    for r in rows:
        st, t = r["stage"], r["now"]
        if st == "event":
            e = r["ev"]
            out.append((t, "trigger", f"② 트리거 B0 → {e['scenario']} {e['cls']} #{e.get('track_id')}"))
        elif st == "suppress":
            out.append((t, "trigger", f"   중복이라 버림 ({r['why']})"))
        elif st == "vlm_input":
            out.append((t, "vlm", f"③ VLM 입력: {r['n_img']}장 {r['img_times'][0]:.1f}–{r['img_times'][-1]:.1f}s "
                                  f"#{r.get('track_id')} 빨간 박스"))
        elif st == "vlm_end":
            lat = (r.get("lat") or {}).get("total_s")
            tag = f"{lat:.2f}s" if lat else "-"
            err = f" 오류 {r['error'][:40]}" if r.get("error") else ""
            out.append((t, "vlm", f"③ VLM 출력 ({tag}): {_fmt_out(r.get('out'))}{err}"))
        elif st == "desc_skip":
            why = "VLM: 위험 아님" if r["why"] == "no_hazard" else r["why"]
            out.append((t, "vlm", f"   설명 안 함 ({why})"))
        elif st == "desc_text":
            out.append((t, "tts", f"④ 하이브리드 → 문장: {r['text']}"))
        elif st == "tts_done":
            how = "캐시" if r.get("cached") else f"합성 {r.get('synth_s', 0):.2f}s"
            out.append((t, "tts", f"⑤ TTS {how}: {r['text']}"))
        elif st == "audio_start":
            k = "경고" if r["kind"] == "warn" else "설명"
            out.append((t, "audio", f"⑥ 재생 [{k}] (트리거 +{t - r['t_event']:.2f}s)"))
        elif st == "audio_drop":
            k = "경고" if r["kind"] == "warn" else "설명"
            out.append((t, "audio", f"⑥ 버림 [{k}] {r['why']}"))
    return out


def render(key, od: Path, fps=15.0, long_side=1280):
    od = Path(od)
    timeline = json.load(open(od / "timeline.json", encoding="utf-8"))
    rows = [json.loads(x) for x in open(od / "log.jsonl", encoding="utf-8")]
    events = [r["ev"] for r in rows if r["stage"] == "event"]
    dets, perc = {}, {}
    for line in open(od / "dets.jsonl", encoding="utf-8"):
        r = json.loads(line)
        dets[r["f"]] = r["d"]
        perc[r["f"]] = r.get("perc_s")
    lines = pipeline_lines(rows)
    args = json.load(open(od / "args.json", encoding="utf-8")) if (od / "args.json").exists() else {}
    vin = {}                                             # 이벤트 → VLM에 실제로 넣은 이미지 (--dump-vlm-inputs)
    for r in rows:
        if r["stage"] == "vlm_input":
            dd = sorted((od / "vlm_inputs").glob(f"ev{r['ev_t']:06.2f}_*"))
            if dd:
                vin[r["now"]] = sorted(dd[0].glob("*.jpg"))
    pw = []                                              # 최근 1 s 인식 시간 (FPS 표시)
    dur = clip_info(key)["dur"]
    write_wav(od / "audio.wav", mix_audio(timeline, dur))

    font = ImageFont.truetype(FONT, 30)
    small = ImageFont.truetype(FONT, 20)
    played = [it for it in timeline if "t_start" in it]
    ff = None
    import imageio_ffmpeg
    for fidx, t, im in iter_frames(key, fps, max_side=1920):
        s = min(1.0, long_side / max(im.shape[:2]))
        hot = {e["track_id"] for e in events if 0 <= t - e["t"] <= 1.5}
        for tid, cls, x1, y1, x2, y2 in dets.get(fidx, []):
            c = (0, 0, 255) if tid in hot else (0, 200, 0)
            cv2.rectangle(im, (x1, y1), (x2, y2), c, 6 if tid in hot else 2)
        im = cv2.resize(im, (round(im.shape[1] * s) // 2 * 2, round(im.shape[0] * s) // 2 * 2))
        vh, vw = im.shape[:2]
        im = np.concatenate([im, np.full((vh, PANEL_W, 3), 18, np.uint8)], axis=1)
        if ff is None:
            H, W = im.shape[:2]
            ff = subprocess.Popen([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
                                   "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-",
                                   "-i", str(od / "audio.wav"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                                   "-crf", "23", "-c:a", "aac", "-b:a", "128k", str(od / "demo.mp4")],
                                  stdin=subprocess.PIPE)
        pil = Image.fromarray(im[:, :, ::-1])
        d = ImageDraw.Draw(pil)
        d.rectangle([0, 0, vw, 34], fill=(0, 0, 0))
        d.text((10, 4), f"{t:5.1f}s   빨간 박스 = 트리거", font=small, fill=(255, 255, 255))
        # ── 오른쪽 패널: 각 단계에 무엇이 들어가고 나오는지 ──
        x0, y = vw + 16, 12
        if perc.get(fidx) is not None:
            pw.append(perc[fidx])
        pw[:] = pw[-int(fps):]
        ms = 1000 * float(np.median(pw)) if pw else 0
        live = args.get("perception") == "yolo"
        d.text((x0, y), "실시간 end-to-end (A5000)" if live else "파이프라인 (저장된 인식 결과)", font=small,
               fill=(255, 255, 255)); y += 34
        d.text((x0, y), "① 인식 YOLO11s-seg(E0) + ByteTrack", font=small, fill=(140, 200, 140)); y += 26
        fps_txt = f"{1000 / ms:4.0f}" if ms else "  -"
        d.text((x0 + 26, y), f"{ms:5.1f} ms/프레임 → {fps_txt} FPS (입력 {fps:g} fps) · 검출 {len(dets.get(fidx, []))}",
               font=small, fill=(220, 220, 220)); y += 30
        vm = args.get("model", "-") if args.get("vlm") == "server" else args.get("vlm", "-")
        tv = f"Supertonic {args.get('tts_voice', '')}" if args.get("tts") == "server" else args.get("tts", "-")
        d.text((x0, y), f"② 트리거 B0 · ③ VLM {vm} · ⑤ TTS {tv}", font=small, fill=(180, 180, 180)); y += 34
        d.line([x0, y, W - 16, y], fill=(80, 80, 80), width=1); y += 10
        col = {"trigger": (255, 200, 90), "vlm": (120, 200, 255), "tts": (200, 160, 255), "audio": (255, 120, 120)}
        for lt, kind, txt in [l for l in lines if l[0] <= t][-14:]:
            c = col[kind] if t - lt < 3 else tuple(int(v * 0.55) for v in col[kind])
            msg = f"{lt:5.1f}s {txt}"
            while d.textlength(msg, font=small) > W - x0 - 10 and len(msg) > 10:
                msg = msg[:-2]
            d.text((x0, y), msg, font=small, fill=c); y += 26
        shown = [k for k in vin if 0 <= t - k < 3.0]
        if shown:                                        # 트리거 출력 → VLM 입력 이미지 그대로
            files = vin[shown[-1]]
            yy = H - 160
            d.text((x0, yy - 28), "VLM에 실제로 들어간 이미지 (과거 → 지금)", font=small, fill=(120, 200, 255))
            tw = (W - x0 - 16) // max(1, len(files))
            for k, fpath in enumerate(files):
                th = Image.open(fpath)
                th.thumbnail((tw - 4, 150))
                pil.paste(th, (x0 + k * tw, yy))
        now = [it for it in played if it["t_start"] <= t < it["t_end"]]
        for k, it in enumerate(now[:1]):
            tag = "경고" if it["kind"] == "warn" else "설명"
            col = (255, 80, 80) if it["kind"] == "warn" else (120, 200, 255)
            txt = f"[{tag}] {it['text']}"
            f = font
            bw = d.textlength(txt, font=f)
            if bw > vw - 30:                                 # 긴 설명은 화면 폭에 맞춰 글자를 줄임
                f = ImageFont.truetype(FONT, max(14, int(30 * (vw - 30) / bw)))
                bw = d.textlength(txt, font=f)
            y = H - 60
            d.rectangle([vw / 2 - bw / 2 - 12, y - 6, vw / 2 + bw / 2 + 12, y + 42], fill=(0, 0, 0))
            d.text((vw / 2 - bw / 2, y), txt, font=f, fill=col)
        ff.stdin.write(np.ascontiguousarray(np.asarray(pil)[:, :, ::-1]).tobytes())
    ff.stdin.close()
    ff.wait()
    print(f"→ {od / 'demo.mp4'}", flush=True)


if __name__ == "__main__":
    render(sys.argv[1], Path(sys.argv[2]) if len(sys.argv) > 2 else OUT / "runtime")
