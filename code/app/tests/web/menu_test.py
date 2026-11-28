"""기능 메뉴 · 점자블록 따라 안내 시험 (서버 없이). 11/24 밤: 점자블록은 별도 화면 없이 같은 화면에서 이어서 안내, 음성은 구글 음성만.
예전 설명:
화면 길게 누르기 → 메뉴 → '버스 언제 와' → 버스 찾기 / '메뉴'라고 말하기 → 메뉴 / "점자블록 찾아줘" → 점자블록 찾기 시작 /
서버가 'tactile_seen' 을 보내면 "점자블록이 있어요. 탐색을 시작할게요." 창 → 6초 뒤 시작, "그만"이면 안 함 / 설정 시트에 점자블록 없음.
실행: python code/app/tests/web/menu_test.py"""
import asyncio, http.server, threading, functools, json, pathlib
from playwright.async_api import async_playwright

WEB = str(pathlib.Path(__file__).resolve().parents[2] / 'web')
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB); H.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()
INIT = """
window.__played=[]; window.__synthUsed=false;
Object.defineProperty(window,'speechSynthesis',{configurable:true,value:{getVoices:()=>[],cancel(){},speak(){window.__synthUsed=true;}}});
class FakeSR { start(){ window.__rec=this; } abort(){} stop(){} }
Object.defineProperty(window,'webkitSpeechRecognition',{configurable:true,value:FakeSR}); Object.defineProperty(window,'SpeechRecognition',{configurable:true,value:FakeSR});
Object.defineProperty(window,'Audio',{configurable:true,value:function(){ const a={_s:'',_t:null,_txt:null,onended:null,onerror:null,volume:1,playbackRate:1,preload:'',muted:false,
 set src(v){a._s=v}, get src(){return a._s},
 play(){ if(!String(a._s).startsWith('blob:')) return Promise.resolve(); const me=a._s; return fetch(me).then(r=>r.text()).then(t=>{ if(a._s!==me) return; a._txt=t; window.__played.push(t); a._t=setTimeout(()=>{ a._t=null;  a._txt=null; a.onended&&a.onended(); }, 40); }); },
 pause(){ if(a._t){ clearTimeout(a._t); a._t=null;  } a._txt=null; } }; return a; }});
"""


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel='chrome', args=['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream'])   # 카메라 · 마이크 허락(가짜 장치)
        c = await b.new_context(permissions=['geolocation'], geolocation={'latitude': 37.45, 'longitude': 126.65, 'accuracy': 5})
        pg = await c.new_page(); pg.on('pageerror', lambda e: print('PAGEERR', e))
        await pg.route('**/api/tts*', lambda r: r.fulfill(status=200, content_type='audio/mpeg', body=__import__('urllib.parse').parse.parse_qs(__import__('urllib.parse').parse.urlparse(r.request.url).query).get('t',[''])[0].encode()))   # 구글 음성 대신: 문장 글자를 그대로 돌려줌
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
        I = "window.__insight"
        res['별도 점자블록 화면 없음'] = await pg.evaluate("!document.querySelector('#tactilePanel') && !window.SayEyeTactile")
        await pg.evaluate(f"{I}.S.pending=[]; {I}.handleSpeech(['점자블록 찾아줘'])"); await asyncio.sleep(0.5)
        res['"점자블록 찾아줘" → 서버에 start'] = await pg.evaluate(f"{I}.S.pending.some(m=>m.type==='tactile_follow'&&m.action==='start')")
        # 서버가 같은 화면에서 이어서 안내 (tactile_follow)
        await pg.evaluate(f"{I}.S.running=true; window.__played=[]; {I}.handle({{type:'tactile_follow', state:'on', key:'found', say:'점자블록이 보여요. 따라가며 안내할게요.', dir:'straight'}})"); await asyncio.sleep(0.2)
        await pg.evaluate(f"{I}.handle({{type:'tactile_follow', state:'on', key:'straight', say:'앞으로 쭉 가세요.', dir:'straight'}})"); await asyncio.sleep(0.2)
        await pg.evaluate(f"{I}.handle({{type:'tactile_follow', state:'on', key:'curve_left', say:'앞에서 점자블록이 왼쪽으로 꺾여요. 천천히 왼쪽으로 트세요.', dir:'curve_left'}})"); await asyncio.sleep(1.5)
        played = await pg.evaluate("window.__played")
        res['따라 안내 음성 (구글 음성 경로)'] = any('따라가며 안내할게요' in x for x in played) and any('앞으로 쭉 가세요' in x for x in played) and any('왼쪽으로 꺾여요' in x for x in played)
        res['같은 화면에 방향 큰 글씨'] = await pg.evaluate("document.querySelector('#sayH').textContent.includes('앞에서 왼쪽으로') && document.querySelector('#askView').hidden && !document.body.classList.contains('tactile-active')")
        await pg.evaluate(f"{I}.S.pending=[]; {I}.handleSpeech(['점자블록 그만'])"); await asyncio.sleep(0.3)
        res['"점자블록 그만" → 서버에 stop'] = await pg.evaluate(f"{I}.S.pending.some(m=>m.type==='tactile_follow'&&m.action==='stop')")
        res['폰 기본 음성 안 씀'] = await pg.evaluate("!window.__synthUsed")
        for k, v in res.items():
            print(('ok  ' if v else 'FAIL'), k)
        print('menu ok' if all(res.values()) else 'menu FAIL')
        await b.close()

asyncio.run(main())
