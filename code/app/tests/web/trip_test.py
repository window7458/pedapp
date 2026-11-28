"""길 안내 + 대중교통 흐름 시험 (서버 없이, 길찾기 응답 · GPS 를 가짜로):
걷기(갈림길 미리 안내 → 정류장 도착) → 511번 버스 기다리기 → "탔어" → 정류장 따라가며 "다음 정류장" · "이번 정류장"
→ "내렸어" → 갈아타러 걷기 → "다 왔어" → 지하철 안내 → "탔어" → "내렸어" → 마지막 걷기 → "다 왔어" → 도착.
실행: python code/app/tests/web/trip_test.py (크롬 · pip install playwright)"""
import asyncio, http.server, threading, functools, json, pathlib
from playwright.async_api import async_playwright

WEB = str(pathlib.Path(__file__).resolve().parents[2] / 'web')
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB); H.log_message = lambda *a: None
srv = http.server.ThreadingHTTPServer(('127.0.0.1', 0), H); threading.Thread(target=srv.serve_forever, daemon=True).start()

O = (37.4500, 126.6560)                                  # 출발
def pt(dn, de): return (O[0] + dn / 111320, O[1] + de / (111320 * 0.7934))   # 북 · 동으로 m 만큼
A = pt(0, 120); TURN = pt(0, 60)                          # 정류장 A (동쪽 120 m), 60 m 지점에서 우회전
STATIONS = [A, pt(0, 600), pt(0, 1100), pt(0, 1600)]       # 511번: A → 정류장 4개
S1 = pt(30, 1650); DEST = pt(30, 3000)
LEG = lambda m, mode, num, a, b, an, bn, stations=(), route='', toward='': dict(
    mode=mode, num=num, route=route or num, toward=toward, minutes=m, meters=500, stops=max(0, len(stations) - 1),
    start=dict(name=an, lat=a[0], lon=a[1]), end=dict(name=bn, lat=b[0], lon=b[1]),
    stations=[dict(name=f'정류장{i}', lat=p[0], lon=p[1]) for i, p in enumerate(stations)])
TRIP = dict(minutes=40, transfers=1, walk_m=300, legs=[
    LEG(2, 'WALK', '', O, A, '출발', '학산정류장'),
    LEG(6, 'BUS', '511', A, STATIONS[-1], '학산정류장', '주안역정류장', STATIONS),
    LEG(1, 'WALK', '', STATIONS[-1], S1, '주안역정류장', '주안'),
    LEG(1, 'SUBWAY', '', S1, DEST, '주안', '용산', [S1, DEST], route='수도권1호선', toward='도화'),
    LEG(1, 'WALK', '', DEST, DEST, '용산', '용산역 광장')])
PLAN = dict(dest=dict(name='용산역 광장', lat=DEST[0], lon=DEST[1]), intro='용산역 광장까지 버스와 지하철로 안내할게요.', mode='transit', trip=TRIP,
            steps=[dict(lat=TURN[0], lon=TURN[1], kind='turn', text='우회전한 뒤 60미터 가세요.'), dict(lat=A[0], lon=A[1], kind='arrive', text='도착했어요.')],
            line=[[O[0], O[1]], [A[0], A[1]]], distance=120, minutes=2, first='동쪽으로 가세요.')
ROUTE = dict(steps=[dict(lat=S1[0], lon=S1[1], kind='arrive', text='도착했어요.')], line=[], distance=60, minutes=1, first='앞으로 가세요.')
INIT = """
window.__played=[];
class FakeSR { start(){ window.__rec=this; } abort(){} stop(){} }
Object.defineProperty(window,'webkitSpeechRecognition',{configurable:true,value:FakeSR}); Object.defineProperty(window,'SpeechRecognition',{configurable:true,value:FakeSR});
Object.defineProperty(window,'Audio',{configurable:true,value:function(){ const a={_s:'',_t:null,_txt:null,onended:null,onerror:null,volume:1,playbackRate:1,preload:'',muted:false,
 set src(v){a._s=v}, get src(){return a._s},
 play(){ if(!String(a._s).startsWith('blob:')) return Promise.resolve(); const me=a._s; return fetch(me).then(r=>r.text()).then(t=>{ if(a._s!==me) return; a._txt=t; window.__played.push(t); a._t=setTimeout(()=>{ a._t=null;  a._txt=null; a.onended&&a.onended(); }, 60); }); },
 pause(){ if(a._t){ clearTimeout(a._t); a._t=null;  } a._txt=null; } }; return a; }});
"""


async def main():
    async with async_playwright() as p:
        b = await p.chromium.launch(channel='chrome', args=['--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream']); c = await b.new_context(permissions=['geolocation'], geolocation={'latitude': O[0], 'longitude': O[1], 'accuracy': 5})
        pg = await c.new_page(); pg.on('pageerror', lambda e: print('PAGEERR', e))
        await pg.route('**/api/tts*', lambda r: r.fulfill(status=200, content_type='audio/mpeg', body=__import__('urllib.parse').parse.parse_qs(__import__('urllib.parse').parse.urlparse(r.request.url).query).get('t',[''])[0].encode()))   # 구글 음성 대신: 문장 글자를 그대로 돌려줌
        await pg.route('**/api/nav/route*', lambda r: r.fulfill(status=200, content_type='application/json', body=json.dumps(ROUTE, ensure_ascii=False)))
        await pg.route('**/api/nav?*', lambda r: r.fulfill(status=200, content_type='application/json', body=json.dumps(PLAN, ensure_ascii=False)))
        await pg.route('**/api/bus*', lambda r: r.fulfill(status=200, content_type='application/json', body=json.dumps(dict(sentence='511번은 3분 뒤에 와요.', min=3, left=2, stop='학산정류장'), ensure_ascii=False)))
        await pg.add_init_script(INIT)
        await pg.goto(f'http://127.0.0.1:{srv.server_port}/index.html', wait_until='domcontentloaded'); await asyncio.sleep(1.5)

        async def go(lat, lon, wait=1.0):
            await c.set_geolocation({'latitude': lat, 'longitude': lon, 'accuracy': 5}); await asyncio.sleep(wait)
        async def hear(t, wait=1.2):
            await pg.evaluate(f"window.__insight.handleSpeech([{json.dumps(t, ensure_ascii=False)}])"); await asyncio.sleep(wait)

        await hear('용산역 광장 가자', 2)
        await go(*pt(0, 15)); await go(*pt(0, 22))              # 갈림길(60 m) 약 40 m 앞 → 미리 안내
        await go(*pt(0, 55)); await go(*pt(0, 112), 2)           # 갈림길 · 정류장 도착 → 버스 기다리기
        await hear('탔어')
        await go(*pt(0, 300), 1); await go(*pt(0, 620)); await go(*pt(0, 1120)); await go(*pt(0, 1500))   # 다음 정류장
        await go(*pt(0, 1590), 1.5)                                                                          # 이번 정류장
        await hear('내렸어', 2)
        await hear('다 왔어', 1.5)
        await hear('탔어'); await hear('내렸어', 2)
        await hear('다 왔어', 1.5)
        said = await pg.evaluate("window.__played")
        for x in said: print(' ·', x)
        need = ['미터 뒤에 우회전이에요. 오른쪽으로 가야 해요', '지금 오른쪽으로 도세요. 그다음 앞으로', '학산정류장에 다 왔어요', '오백십일번 버스를 기다릴게요', '다음 정류장 주안역정류장에서 내려요',
                '이번 정류장 주안역정류장에서 내리세요', '주안까지 약', '주안역에서 수도권1호선 도화 방향', '용산역에서 내려요', '용산역 광장에 도착했어요']
        miss = [n for n in need if not any(n in x for x in said)]
        print('MISSING', miss) if miss else print('trip ok')
        await b.close()

asyncio.run(main())
