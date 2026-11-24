"""서버 음성 인식 끝까지: 가짜 마이크(wav, 16 kHz 모노, 앞에 8초 정도 여유) → Silero VAD → /api/stt(이 PC 의 실제 Whisper) → 명령."""
# 실행: python code/app/tests/web/stt_e2e_test.py 녹음.wav (크롬 설치 필요 · pip install playwright)
# 서버 없이 화면(web/index.html)만 띄워 시험한다. 웹소켓 연결 실패 기록은 정상.
"""폰과 같은 흐름: 가짜 마이크(wav) → 앱(earsSrv, 말 구간 자르기) → /api/stt(실제 Whisper, 이 PC CPU) → handleSpeech → 명령."""
import asyncio, http.server, threading, functools, sys, wave, json, pathlib, numpy as np
APP = pathlib.Path(__file__).resolve().parents[2]; sys.path.insert(0, str(APP)); import stt
from playwright.async_api import async_playwright
WEB, WAV = str(APP / 'web'), sys.argv[1]
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB); H.log_message=lambda *a: None
srv = http.server.ThreadingHTTPServer(('127.0.0.1',0),H); threading.Thread(target=srv.serve_forever,daemon=True).start()
INIT = """window.__cmds=[]; window.__heard=[];"""
async def main():
    stt.load()
    async with async_playwright() as p:
        b = await p.chromium.launch(channel='chrome', args=['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream',f'--use-file-for-fake-audio-capture={WAV}%noloop','--autoplay-policy=no-user-gesture-required'])
        c = await b.new_context(permissions=['microphone']); pg = await c.new_page()
        pg.on('console', lambda m: print('CONSOLE', m.text[:200]) if (('ears' in m.text or 'rror' in m.text or 'vad' in m.text.lower()) and 'WebSocket' not in m.text) else None)
        async def stt_route(route):
            body = route.request.post_data_buffer or b''
            if route.request.method == 'GET':
                return await route.fulfill(status=200, content_type='application/json', body='{"on":true,"ready":true}')
            r = await asyncio.to_thread(stt.transcribe, body)
            print('STT', round(len(body)/32000,2), 's ->', r)
            await route.fulfill(status=200, content_type='application/json', body=json.dumps(r, ensure_ascii=False))
        await pg.route('**/api/stt', stt_route)
        await pg.route('**/api/bus*', lambda r: r.fulfill(status=200, content_type='application/json', body='{"sentence":"511번 버스는 3분 뒤에 와요."}'))
        await pg.add_init_script(INIT + "Object.defineProperty(window,'webkitSpeechRecognition',{value:undefined}); Object.defineProperty(window,'SpeechRecognition',{value:undefined});")  # 서버 인식(earsSrv) 경로를 시험
        await pg.goto(f'http://127.0.0.1:{srv.server_port}/index.html', wait_until='domcontentloaded')
        await pg.evaluate("()=>{ const P=window.__insight.parseCommand; window.__insight.S.geo={latitude:37.45,longitude:126.65,accuracy:10}; }")
        await pg.mouse.click(200, 300)
        for i in range(70):
            await asyncio.sleep(1)
            ear = await pg.evaluate("document.querySelector('#earText').textContent")
            if '들은 말' in ear: break
        print('ENGINE', await pg.evaluate("ears.engine||'energy'"), 'ORT', await pg.evaluate('!!window.ort'), 'VAD', await pg.evaluate('!!window.vad')); print('EAR', ear, '| mode', await pg.evaluate("window.__insight && typeof ears!=='undefined' ? ears.mode() : '?'"))
        print('latest', await pg.evaluate("window.__insight.S.busRoute"), await pg.evaluate("window.__insight.S.navQuery"))
        await b.close()
asyncio.run(main())
