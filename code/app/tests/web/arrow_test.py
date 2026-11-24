"""미리보기 걷는 방향 화살표 그림: arrow_small.png · arrow_big.png 를 만든다 (눈으로 확인)."""
# 실행: python code/app/tests/web/arrow_test.py [저장 폴더] (크롬 설치 필요 · pip install playwright)
# 서버 없이 화면(web/index.html)만 띄워 시험한다. 웹소켓 연결 실패 기록은 정상.
import asyncio, http.server, threading, functools, sys, pathlib
from playwright.async_api import async_playwright
WEB=str(pathlib.Path(__file__).resolve().parents[2] / 'web'); OUT=sys.argv[1] if len(sys.argv)>1 else '.'
H=functools.partial(http.server.SimpleHTTPRequestHandler, directory=WEB); H.log_message=lambda *a: None
srv=http.server.ThreadingHTTPServer(('127.0.0.1',0),H); threading.Thread(target=srv.serve_forever,daemon=True).start()
async def main():
    async with async_playwright() as p:
        b=await p.chromium.launch(channel='chrome'); pg=await b.new_page(viewport={'width':412,'height':860}, device_scale_factor=2)
        await pg.goto(f'http://127.0.0.1:{srv.server_port}/index.html', wait_until='domcontentloaded'); await asyncio.sleep(7)
        await pg.evaluate("""()=>{ const c=document.createElement('canvas'); c.width=720; c.height=960; const x=c.getContext('2d');
          const g=x.createLinearGradient(0,0,0,960); g.addColorStop(0,'#9db4c8'); g.addColorStop(.45,'#c8c2b4'); g.addColorStop(1,'#7b766c'); x.fillStyle=g; x.fillRect(0,0,720,960);
          const im=document.querySelector('#pv'); im.src=c.toDataURL(); im.hidden=false; document.body.classList.add('hasvideo');
          const S=window.__insight.S; S.running=true; S.prev=true; S.saver=false; S.wsOk=true; document.querySelector('#stage').hidden=false;
          setInterval(()=>{ S.wsOk=true; S.dets={W:720,H:960,d:[[1,'person',.55,.3,.7,.7]], overlay:{path:{valid:true, polygon:[[.30,1.0],[.70,1.0],[.56,.42],[.48,.42]]}}}; S.detsAt=performance.now(); },200); }""")
        await asyncio.sleep(1)
        await pg.screenshot(path=OUT+'/arrow_small.png')
        await pg.evaluate("document.querySelector('#stage').classList.add('big')"); await asyncio.sleep(0.6)
        print('px', await pg.evaluate("(()=>{const ov=document.querySelector('#ov'), c=ov.getContext('2d'); let n=0; const d=c.getImageData(0,0,ov.width,ov.height).data; for(let i=0;i<d.length;i+=4){ if(d[i+3]>0) n++; } return n })()"))
        print('cs', await pg.evaluate("getComputedStyle(document.querySelector('#ov')).display + ' ' + getComputedStyle(document.querySelector('#ov')).zIndex + ' ' + getComputedStyle(document.querySelector('#pv')).zIndex"))
        await pg.screenshot(path=OUT+'/arrow_big.png')
        await b.close()
asyncio.run(main())
