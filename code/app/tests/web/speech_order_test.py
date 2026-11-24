"""말 순서 시험: 대답 중 보통 경고 · 긴급 경고 · 사용자가 말하는 중 경고 · 버스 물을 때 경고 멈춤 · 번호 읽기."""
# 실행: python code/app/tests/web/speech_order_test.py (크롬 설치 필요 · pip install playwright)
# 서버 없이 화면(web/index.html)만 띄워 시험한다. 웹소켓 연결 실패 기록은 정상.
import asyncio, http.server, threading, functools, sys, json, pathlib
from playwright.async_api import async_playwright
WEB = str(pathlib.Path(__file__).resolve().parents[2] / 'web')
H = functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB); H.log_message=lambda *a: None
srv = http.server.ThreadingHTTPServer(('127.0.0.1',0),H); threading.Thread(target=srv.serve_forever,daemon=True).start()
UA='Mozilla/5.0 (Linux; Android 17; SM-S926N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/157.0 Mobile Safari/537.36'
INIT = """
window.__played=[];
class FakeSR { start(){ window.__rec=this; } abort(){} stop(){} }
Object.defineProperty(window,'webkitSpeechRecognition',{configurable:true,value:FakeSR}); Object.defineProperty(window,'SpeechRecognition',{configurable:true,value:FakeSR});
Object.defineProperty(window,'Audio',{configurable:true,value:function(){ return {play:()=>Promise.reject(new Error('x')), pause(){}, set src(v){}, get src(){return ''}}; }});
// 음성: /api/tts 대신 폰 음성으로 떨어지게 하고, 말한 문장을 기록 (1글자=30ms)
Object.defineProperty(window,'speechSynthesis',{configurable:true,value:{ getVoices:()=>[], cancel(){ if(window.__u){ clearTimeout(window.__u.t); window.__played.push('[끊김] '+window.__u.text); window.__u=null; } },
  speak(u){ if(u.text.trim()==='') return; window.__u=u; u.t=setTimeout(()=>{ window.__played.push(u.text); window.__u=null; u.onend&&u.onend(); }, u.text.length*30); } }});
window.fetch = ((f)=> (url, o)=> String(url).startsWith('/api/tts') ? Promise.resolve(new Response(null,{status:404})) : f(url,o))(window.fetch);
"""
async def main():
  async with async_playwright() as p:
    b=await p.chromium.launch(channel='chrome'); c=await b.new_context(user_agent=UA); pg=await c.new_page(); pg.on('pageerror', lambda e: print('PAGEERR', e)); None
    await c.grant_permissions(['geolocation']); await c.set_geolocation({'latitude':37.45,'longitude':126.65}); await pg.add_init_script(INIT); await pg.goto(f'http://127.0.0.1:{srv.server_port}/index.html', wait_until='domcontentloaded')
    await asyncio.sleep(1.5)
    r = await pg.evaluate("""()=>{ const P=window.__insight.parseCommand, out={};
      for(const t of ['511 버스 언제 와','511번 버스 언제 와','오백십일번 버스 언제와','버스 언제 와','인하대역 가자']){ const c=P(t); out[t]=[c, window.__insight.S.busRoute, window.__insight.S.navQuery]; window.__insight.S.busRoute=undefined; }
      return out; }""")
    print(json.dumps(r, ensure_ascii=False))
    await pg.mouse.click(200, 300); await asyncio.sleep(0.3)
    # 1) 대답 중 보통(1) 경고 → 대답을 끊지 않음
    await pg.evaluate("()=>{ window.__played=[]; say('511번 버스는 3분 뒤에 와요. 두 정거장 전이에요.'); setTimeout(()=>window.__insight.handle({type:'warn', text:'12시 방향, 사람에 유의하세요.', prio:1}), 400); }")
    await asyncio.sleep(4); print('case1', await pg.evaluate("window.__played"))
    # 2) 대답 중 긴급(3) 경고 → 끊고 경고, 대답 다시
    await pg.evaluate("()=>{ window.__played=[]; say('511번 버스는 3분 뒤에 와요. 두 정거장 전이에요.'); setTimeout(()=>window.__insight.handle({type:'warn', text:'멈추세요.', prio:3}), 400); }")
    await asyncio.sleep(4); print('case2', await pg.evaluate("window.__played"))
    print("ear:", await pg.evaluate("document.querySelector('#earText').textContent"), await pg.evaluate("!!window.__rec"))
    # 3) 사용자가 말하는 중(중간 결과) 보통 경고 → 말 끝날 때까지 기다림
    await pg.evaluate("""()=>{ window.__played=[]; const R=window.__rec;
      const res=(t,fin)=>{ const a=[{transcript:t}]; a.isFinal=fin; return a; };
      R.onresult({resultIndex:0, results:[res('511 버스',false)]});
      window.__insight.handle({type:'warn', text:'12시 방향, 사람에 유의하세요.', prio:1});
      setTimeout(()=>{ window.__t1=window.__played.length; R.onresult({resultIndex:0, results:[res('511 버스 언제 와',true)]}); }, 1000); }""")
    await asyncio.sleep(3); print('case3 played-before-final:', await pg.evaluate("window.__t1"), await pg.evaluate("window.__played"))
    # 4) 버스 물을 때: 대답 끝까지 경고 안 함, 끝난 뒤엔 다시 경고
    await pg.route('**/api/bus*', lambda r: r.fulfill(status=200, content_type='application/json', body='{"sentence":"511번 버스는 3분 뒤에 와요. 두 정거장 전이에요."}'))
    await pg.evaluate("()=>{ window.__played=[]; window.__insight.S.geo={latitude:37.45,longitude:126.65,accuracy:10}; window.__insight.handleSpeech(['511 버스 언제 와']); setTimeout(()=>window.__insight.handle({type:'warn', text:'12시 방향, 사람에 유의하세요.', prio:1}), 800); }")
    for _ in range(30):
        await asyncio.sleep(0.5)
        if not await pg.evaluate("INFO.on"): break
    await pg.evaluate("()=>window.__insight.handle({type:'warn', text:'1시 방향, 자전거에 유의하세요.', prio:1})")
    await asyncio.sleep(2); print('case4', await pg.evaluate("window.__played"))
    #print('dbg', await pg.evaluate("({latest: window.__insight.S.latest, info: INFO.on, talking: ears.talking(), mode: ears.mode(), busy: speech.busy(), idle: speech.idle()})"))
    await b.close()
asyncio.run(main())
