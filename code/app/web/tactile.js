/* SAYEYE_TACTILE_V6: start once; automatic snapshots; audio is optional. */
window.SayEyeTactile=(()=>{
  let selected=false,active=false,ack=false,starting=false,session='',epoch=0;
  let last=null,maskImage=null,version=0,lastAt=0;
  const SAMPLE_DELAY=500, MAX_AGE=3000, RESPONSE_TIMEOUT=8000;
  const GUIDE_INTERVAL=3000, SEARCH_INTERVAL=1000;
  let suspended=false,pauseInfo=null,retries=0,retryAfterStart=false;
  let pending=null,nextCapture=0;
  const captureCanvas=document.createElement('canvas'),captureCtx=captureCanvas.getContext('2d');
  let spokenAt=-1e9,lastSpoken='',staleSaid=false,saved=null;
  const panel=document.querySelector('#tactilePanel'),text=document.querySelector('#tactileText');
  const title=document.querySelector('#tactileHeading'),stats=document.querySelector('#tactileStats');
  const phaseText=document.querySelector('#tactilePhase');
  const confirmButton=document.querySelector('#tactileConfirm'),pauseButton=document.querySelector('#tactileStop');
  const restartButton=document.querySelector('#tactileRestart'),stage=document.querySelector('#stage');
  const anchor=document.createComment('camera home');stage.parentNode.insertBefore(anchor,stage);
  document.body.appendChild(panel);
  function show(heading,body){panel.hidden=false;title.textContent=heading;text.textContent=body;}
  function clear(){last=null;maskImage=null;version++;confirmButton.hidden=true;stats.textContent='';}
  function resetCapture(){pending=null;nextCapture=performance.now()+SAMPLE_DELAY;}
  function hideMask(){last=null;maskImage=null;version++;confirmButton.hidden=true;}
  function newSession(){return crypto.randomUUID?crypto.randomUUID():Date.now().toString(36)+Math.random().toString(36).slice(2);}
  function queueStart(){
    // One pending request for the existing app's reconnect handler to flush.
    S.pending=S.pending.filter(m=>m.type!=='tactile');
    S.pending.push({type:'tactile',action:'start',session});
  }
  function retrySession(message){
    if(!selected||!active||suspended||document.hidden)return;
    if(starting){retryAfterStart=true;return;}
    epoch++;resetCapture();ack=false;hideMask();speech.dropTactile();session=newSession();
    lastAt=performance.now();staleSaid=false;retries++;
    S.pending=S.pending.filter(m=>m.type!=='tactile');
    if(S.ws&&S.ws.readyState===1)sendCmd({type:'tactile',action:'start',session});else queueStart();
    show('자동으로 다시 확인하는 중',message);
    phaseText.textContent=`자동 재시도 ${retries}회 · 다시 시작을 누르지 않아도 돼요.`;
  }
  function controls(){pauseButton.hidden=!active;restartButton.hidden=active;}
  function fresh(m){
    if(!m || !Number.isFinite(m.capture_ms))return false;
    const age=Date.now()-m.capture_ms;
    return age>=-500 && age<MAX_AGE && performance.now()-lastAt<MAX_AGE;
  }
  function speak(message,key,force=false){
    if(!selected || !message)return;
    const now=performance.now();
    if(!force&&key===lastSpoken&&now-spokenAt<8000)return;
    // Do not keep restarting a long identical sentence. Speech never gates capture.
    if(key===lastSpoken&&speech.kind&&speech.kind()==='tactile'&&speech.busy&&speech.busy())return;
    try{speech.dropTactile();speech.push('tactile',message);}catch(e){}
    lastSpoken=key;spokenAt=now;
  }
  function repeat(){
    if(active){
      if(suspended)return;
      if(!pending){hideMask();nextCapture=performance.now()+SAMPLE_DELAY;}
      show('새 사진으로 확인하는 중','자동 안내가 계속되고 있어요.');return;
    }
    speak(text.textContent,'repeat',true);
  }
  function showPause(){
    if(!pauseInfo)return;
    show(pauseInfo.reason==='complete'?'도착 확인 완료':pauseInfo.reason==='error'?'분석 오류':'찾기 중지',pauseInfo.message);
    phaseText.textContent='중지 원인: '+pauseInfo.reason;stats.textContent=pauseInfo.stats;
  }
  function pause(message='사용자가 점자블록 찾기를 중지했어요.',remote=false,reason='중지 버튼'){
    if(!selected)return;
    if(!active&&pauseInfo&&!remote){showPause();return;}
    pauseInfo={message,reason,stats:stats.textContent};suspended=false;retryAfterStart=false;
    active=false;ack=false;epoch++;resetCapture();clear();speech.dropTactile();controls();
    S.pending=S.pending.filter(m=>m.type!=='tactile');
    if(!remote)sendCmd({type:'tactile',action:'pause',session});
    showPause();
  }
  function leave(){
    selected=false;active=false;ack=false;starting=false;epoch++;resetCapture();clear();speech.stop();
    S.pending=S.pending.filter(m=>m.type!=='tactile');
    anchor.parentNode.insertBefore(stage,anchor.nextSibling);
    document.body.classList.remove('tactile-active');panel.hidden=true;
    if(saved){S.prev=saved.prev;S.saver=saved.saver;saved=null;applySettings();save();}
    S.dets=null;S.detsAt=0;S.spokeReady=false;S.stallSaid=false;S.latest='';S.lastWarns=[];
    document.body.classList.remove('warn','caut');notice('');setGo();calmView();
    document.querySelector('#openSheet').focus();session='';suspended=false;pauseInfo=null;retryAfterStart=false;
  }
  function back(){
    if(!selected)return;
    active=false;ack=false;epoch++;resetCapture();clear();speech.dropTactile();controls();
    if(S.ws&&S.ws.readyState===1){
      show('기본 안내로 전환 중','잠시만 기다려 주세요.');
      sendCmd({type:'tactile',action:'stop',session});
    }else{stop();leave();}
  }
  async function begin(){
    if(starting||active)return;
    starting=true;speech.unlock();speech.stop();openSheet(false);wake();
    if(!selected){
      selected=true;epoch++;saved={prev:S.prev,saver:S.saver};
      document.querySelector('#tactileCamera').appendChild(stage);
      document.body.classList.add('tactile-active');
      S.prev=true;S.saver=false;applySettings();
    }
    session=newSession();suspended=false;pauseInfo=null;retries=0;retryAfterStart=false;
    const own=session;
    active=true;ack=false;epoch++;resetCapture();lastAt=performance.now();lastSpoken='';staleSaid=false;
    clear();controls();S.dets=null;S.lastWarns=[];
    show('자동으로 점자블록 찾는 중','휴대폰을 몸 정면에 맞춰 주세요. 새 사진으로 방향을 계속 갱신해요.');title.focus();
    stats.textContent='자동 안내 V6 · 촬영과 다음 안내는 자동으로 반복';phaseText.textContent='음성 재생 여부와 관계없이 진행해요.';
    sendCmd({type:'tactile',action:'start',session});
    try{
      if(!S.running)await start();
      if(session!==own||!selected)return;
      if(!S.running){pause('카메라를 켜지 못했어요. 권한을 허용한 뒤 다시 찾기를 눌러 주세요.',false,'카메라 권한/시작 실패');return;}
      setGo();speak('점자블록 찾기를 시작합니다. 멈춘 상태에서 앞쪽 바닥을 비춰 주세요.','start');
    }finally{
      starting=false;
      if(retryAfterStart&&selected&&active&&S.running&&!suspended&&!document.hidden){
        retryAfterStart=false;retrySession('카메라 준비가 끝나 자동 안내를 이어가요.');
      }
    }
  }
  function handle(m){
    if(!selected||m.session!==session)return;
    if(m.control){
      if(m.state==='off'){leave();say(S.running?'기본 위험 안내를 다시 시작합니다.':'기본 화면으로 돌아왔어요.');return;}
      if(m.state==='searching'){if(active&&!suspended&&!ack){ack=true;nextCapture=performance.now()+SAMPLE_DELAY;}return;}
      if(m.state==='confirm_wait'){clear();speech.dropTactile();show('가까움을 다시 확인하는 중',m.text);return;}
      if(['paused','complete','error'].includes(m.state)){
        if(m.state==='paused'){
          if(suspended)return;
          if(!active&&pauseInfo){showPause();return;}
          if(active){retrySession('서버 상태를 맞춘 뒤 자동으로 다시 분석해요.');return;}
        }
        pause(m.text,true,m.state);title.textContent=m.state==='complete'?'도착 확인 완료':m.state==='error'?'분석을 중지했어요':'찾기 중지';
        speak(m.text,m.state,true);return;
      }
      return;
    }
    if(!active||suspended||!ack||!pending||!pending.sent||m.capture_ms!==pending.cap)return;
    // A late/duplicate reply cannot release a different frame's reservation.
    const roundTrip=Math.round(performance.now()-pending.at);
    pending=null;nextCapture=performance.now()+SAMPLE_DELAY;
    lastAt=performance.now();clear();
    const age=Date.now()-m.capture_ms;
    stats.textContent=`자동 안내 V6 · 분석 ${m.processing_ms||0}ms · 왕복 ${roundTrip}ms · 결과 지연 ${Number.isFinite(age)?Math.max(0,age):'?'}ms${m.confidence?' · 검출 점수 '+Math.round(m.confidence*100)+'%':''}`;
    if(!fresh(m)){
      show('최신 사진으로 다시 확인','늦게 도착한 결과예요. 새 사진을 자동으로 다시 분석해요.');phaseText.textContent='자동 안내 유지 · 다시 시작할 필요 없어요.';
      speak('분석이 늦어요. 멈춘 상태에서 기다려 주세요.','stale');return;
    }
    last=m;staleSaid=false;
    const heading={searching:'점자블록 찾는 중',lost:'점자블록을 다시 찾는 중',candidate:'점자블록 후보',detected:'검출 영역 확인 중',acquiring:'점자블록 확인 중',approach:'점자블록 방향',near_check:'가까움을 확인하는 중',near:'멈추고 직접 확인하세요',stale:'분석을 기다리는 중'}[m.state]||'점자블록 찾기';
    show(heading,m.text||'');
    confirmButton.hidden=m.state!=='near';
    if(m.mask_png&&m.mask_png.startsWith('data:image/png;base64,')){
      const rev=version,img=new Image();img.onload=()=>{if(active&&rev===version&&fresh(m))maskImage=img;};img.src=m.mask_png;
    }
    const action=m.state==='approach'&&['turn_left','turn_right','forward_step'].includes(m.action);
    nextCapture=performance.now()+(action?GUIDE_INTERVAL:(m.state==='near_check'||m.state==='near'||m.state==='stale')?SAMPLE_DELAY:SEARCH_INTERVAL);
    phaseText.textContent=action?'안내를 따라 이동하면 약 3초 뒤 새 사진으로 방향을 갱신해요.':'자동으로 다음 사진을 확인해요.';
    const key=m.state+':'+(m.direction||'');speak(m.text,key,action);
  }
  function draw(canvas,ctx,r){
    if(!selected||!active||suspended||!S.running||!S.wsOk||!fresh(last)||!maskImage)return;
    const v=document.querySelector('#cam');
    if(v.hidden||!v.videoWidth||!last.W||!last.H||Math.abs((v.videoWidth/v.videoHeight)/(last.W/last.H)-1)>.02)return;
    const k=Math.min(r.width/v.videoWidth,r.height/v.videoHeight),w=v.videoWidth*k,h=v.videoHeight*k,ox=(r.width-w)/2,oy=(r.height-h)/2;
    ctx.save();ctx.beginPath();ctx.rect(ox,oy,w,h);ctx.clip();
    ctx.globalAlpha=last.state==='candidate'?.45:1;ctx.drawImage(maskImage,ox,oy,w,h);ctx.globalAlpha=1;
    if(last.target&&last.state==='approach'){
      const [x,y]=last.target;if(Number.isFinite(x)&&Number.isFinite(y)){
        ctx.beginPath();ctx.arc(ox+x*w,oy+y*h,8,0,Math.PI*2);ctx.lineWidth=3;ctx.strokeStyle='#fff';ctx.stroke();
      }
    }
    ctx.restore();
  }
  function allowFrame(){return !selected;} // Tactile uses sendFrame, never the normal sender.
  function sendFrame(video){
    if(!selected||!active||suspended||document.hidden||!ack||!S.running||S.mode!=='phone'||pending||performance.now()<nextCapture)return;
    const socket=S.ws;
    if(!socket||socket.readyState!==1||socket.bufferedAmount>0||!video.videoWidth||!video.videoHeight)return;
    if(Number.isFinite(video.readyState)&&video.readyState<2)return;
    const frame={epoch,cap:Date.now(),at:performance.now(),socket,sent:false};
    pending=frame; // Reserve before asynchronous JPEG encoding.
    phaseText.textContent='새 사진 분석 중 · 다음 안내도 자동으로 이어져요.';
    const valid=()=>pending===frame&&epoch===frame.epoch&&selected&&active&&ack&&S.running&&S.ws===socket&&socket.readyState===1;
    const retry=()=>{if(pending===frame){pending=null;nextCapture=performance.now()+SAMPLE_DELAY;}};
    try{
      const scale=Math.min(1,960/Math.max(video.videoWidth,video.videoHeight));
      captureCanvas.width=Math.round(video.videoWidth*scale);captureCanvas.height=Math.round(video.videoHeight*scale);
      captureCtx.drawImage(video,0,0,captureCanvas.width,captureCanvas.height);
      captureCanvas.toBlob(async blob=>{
        try{
          if(!blob||!valid()){retry();return;}
          const buffer=await blob.arrayBuffer();
          if(!valid()){retry();return;}
          // Discard a frame delayed during encoding; take a new picture later.
          if(performance.now()-frame.at>=MAX_AGE||socket.bufferedAmount>0){retry();return;}
          const out=new Uint8Array(12+buffer.byteLength),dv=new DataView(out.buffer);
          out.set([83,69,70,49]);dv.setFloat64(4,frame.cap,false);out.set(new Uint8Array(buffer),12);
          frame.sent=true;socket.send(out.buffer);
        }catch(error){retry();}
      },'image/jpeg',.85);
    }catch(error){retry();}
  }
  function connectionLost(){
    if(!selected||!active)return;
    // Do not invalidate a camera permission request that is still starting.
    if(starting)retryAfterStart=true;else{epoch++;session=newSession();}
    resetCapture();ack=false;hideMask();speech.dropTactile();lastAt=performance.now();
    if(!suspended)queueStart();
    show('연결을 기다리는 중','연결이 돌아오면 자동으로 점자블록 안내를 재개해요.');
    phaseText.textContent='자동 재연결 중 · 오래된 방향 안내는 사용하지 않아요.';
  }
  function suspend(){
    if(!selected||!active||suspended)return;
    suspended=true;if(!starting)epoch++;resetCapture();ack=false;hideMask();speech.dropTactile();
    S.pending=S.pending.filter(m=>m.type!=='tactile');
    if(S.ws&&S.ws.readyState===1)sendCmd({type:'tactile',action:'pause',session});
    show('화면이 돌아오면 계속해요','화면이 가려져 촬영을 쉬고 있어요. 돌아오면 자동으로 재개해요.');
  }
  function resume(){
    if(!selected||!active||!suspended||document.hidden)return;
    suspended=false;retrySession('현재 화면에서 자동 안내를 다시 이어가요.');
  }
  document.querySelector('#tactileStart').addEventListener('click',begin);
  pauseButton.addEventListener('click',()=>{pause();repeat();});
  restartButton.addEventListener('click',begin);
  document.querySelector('#tactileBack').addEventListener('click',back);
  document.querySelector('#tactileRepeat').addEventListener('click',repeat);
  confirmButton.addEventListener('click',()=>{
    if(active&&last&&last.state==='near'&&fresh(last)){speech.dropTactile();sendCmd({type:'tactile',action:'confirm',session});}
  });
  setInterval(()=>{
    if(!selected||!active||starting||suspended||document.hidden)return;
    if(pending&&performance.now()-pending.at>RESPONSE_TIMEOUT){
      retrySession('응답이 늦어 새 사진으로 자동 재시도해요. 최신 안내를 기다려 주세요.');return;
    }
    if(last&&!fresh(last)){hideMask();show('다음 안내 준비 중','새 사진으로 방향을 갱신하고 있어요.');}
    if(!pending&&performance.now()-lastAt>RESPONSE_TIMEOUT&&S.ws&&S.ws.readyState===1){
      retrySession('연결 상태를 맞춘 뒤 자동으로 다시 확인해요.');
    }
  },250);
  addEventListener('pagehide',suspend);addEventListener('pageshow',resume);
  document.addEventListener('visibilitychange',()=>{if(document.hidden)suspend();else resume();});
  return {begin,handle,draw,repeat,back,sendFrame,selected:()=>selected,allowFrame,frameToken:()=>epoch,
    validFrame:token=>token===epoch&&(!selected||(active&&ack&&!suspended)),stop:()=>pause('앱 안내가 중지되어 점자블록 찾기도 멈췄어요.',false,'앱 안내 중지'),connectionLost};
})();
