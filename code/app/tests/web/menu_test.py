"""기능 메뉴 · 점자블록 시험 (서버 없이):
화면 길게 누르기 → 메뉴 → '버스 언제 와' → 버스 찾기 / '메뉴'라고 말하기 → 메뉴 / "점자블록 찾아줘" → 점자블록 찾기 시작 /
서버가 'tactile_seen' 을 보내면 "점자블록이 있어요. 탐색을 시작할게요." 창 → 6초 뒤 시작, "그만"이면 안 함 / 설정 시트에 점자블록 없음.
실행: python code/app/tests/web/menu_test.py"""
import asyncio, http.server, threading, functools, json, pathlib
from playwright.async_api import async_playwright

WEB = str(pathlib.Path(__file__).resolve().parents[2] / 'web')
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB); H.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
INIT = """
window.__played=[];
class FakeSR { start(){ window.__rec=this; } abort(){} stop(){} }
Object.defineProperty(window,'webkitSpeechRecognition',{configurable:true,value:FakeSR}); Object.defineProperty(window,'SpeechRecognition',{configurable:true,value:FakeSR});
Object.defineProperty(window,'Audio',{configurable:true,value:function(){ return {play:()=>Promise.reject(new Error('x')), pause(){}, set src(v){}, get src(){return ''}}; }});
Object.defineProperty(window,'speechSynthesis',{configurable:true,value:{ getVoices:()=>[], cancel(){ if(window.__u){ clearTimeout(window.__u.t); window.__u=null; } },
  speak(u){ if(u.text.trim()==='') return; window.__u=u; window.__played.push(u.text); u.t=setTimeout(()=>{ window.__u=null; u.onend&&u.onend(); }, 40); } }});
"""


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel='chrome')
        c = await b.new_context(permissions=['geolocation'], geolocation={'latitude': 37.45, 'longitude': 126.65, 'accuracy': 5})
        pg = await c.new_page(); pg.on('pageerror', lambda e: print('PAGEERR', e))
        await pg.route('**/api/tts*', lambda r: r.fulfill(status=404, body=''))
        await pg.route('**/api/bus*', lambda r: r.fulfill(status=200, content_type='application/json', body=json.dumps(dict(sentence='학산정류장에 511번이 3분 뒤에 와요.'), ensure_ascii=False)))
        await pg.add_init_script(INIT)
        await pg.goto(f'http://127.0.0.1:{srv.server_port}/index.html', wait_until='domcontentloaded'); await asyncio.sleep(7)   # 로딩 화면이 걷힌 뒤
        res = {}
        res['설정에 점자블록 없음'] = await pg.evaluate("!document.querySelector('#sheet #tactileStart')")
        await pg.mouse.move(200, 300); await pg.mouse.down(); await asyncio.sleep(0.8); await pg.mouse.up(); await asyncio.sleep(0.4)
        res['길게 누르면 메뉴'] = await pg.evaluate("!document.querySelector('#menuView').hidden")
        await pg.click('#menuView .mbtn[data-m=bus]'); await asyncio.sleep(2.5)
        played = await pg.evaluate("window.__played")
        res['메뉴 버스 → 버스 찾기'] = any('정류장의 버스를 찾아볼게요' in x for x in played) and any('3분 뒤에 와요' in x for x in played)
        res['메뉴 닫힘'] = await pg.evaluate("document.querySelector('#menuView').hidden")
        await pg.evaluate("window.__insight.handleSpeech(['메뉴'])"); await asyncio.sleep(0.4)
        res['"메뉴" 말하면 메뉴'] = await pg.evaluate("!document.querySelector('#menuView').hidden")
        await pg.click('#menuClose'); await asyncio.sleep(0.3)
        sel = "window.SayEyeTactile.selected()"
        await pg.evaluate("window.__insight.handleSpeech(['점자블록 찾아줘'])"); await asyncio.sleep(1)
        res['"점자블록 찾아줘" → 시작'] = await pg.evaluate(sel)
        await pg.evaluate("window.SayEyeTactile.back()"); await asyncio.sleep(1)
        res['기본 안내로 돌아옴'] = not await pg.evaluate(sel)
        await pg.evaluate("window.__insight.S.running=true; window.__played=[]; window.__insight.handle({type:'tactile_seen', area:0.03})"); await asyncio.sleep(0.6)
        res['점자블록 보임 → 안내 창'] = await pg.evaluate("!document.querySelector('#askView').hidden && document.querySelector('#askH').textContent==='점자블록이 있어요'")
        res['점자블록 보임 → 음성'] = any('점자블록이 있어요. 탐색을 시작할게요' in x for x in await pg.evaluate("window.__played"))
        await asyncio.sleep(6.5)
        res['6초 뒤 자동 시작'] = await pg.evaluate(sel)
        await pg.evaluate("window.SayEyeTactile.back()"); await asyncio.sleep(1)
        await pg.evaluate("window.__insight.S.running=true; window.__insight.handle({type:'tactile_seen', area:0.03})"); await asyncio.sleep(0.5)
        await pg.evaluate("window.__insight.handleSpeech(['그만'])"); await asyncio.sleep(7)
        res['"그만"이면 시작 안 함'] = not await pg.evaluate(sel)
        for k, v in res.items():
            print(('ok  ' if v else 'FAIL'), k)
        print('menu ok' if all(res.values()) else 'menu FAIL')
        await b.close()

asyncio.run(main())
