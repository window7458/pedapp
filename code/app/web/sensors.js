/* 폰 센서 수집: 앱을 켜면 자동으로 시작해서 서버로 보낸다.
   GPS · 자이로 · 가속도 · 중력 · 방향 · 나침반. (걸음 수 · 활동 · 기압계는 웹 브라우저로 읽을 수 없음)
   아이폰은 자이로 권한을 '화면을 한 번 누른 순간'에만 물을 수 있어서, 켜자마자 GPS를 시작하고 첫 터치에 자이로를 켠다. */
window.SayEyeSensors = (() => {
  const B = { motion: [], orient: [], gps: [] }, MAX = 2000;
  const st = { motion: 'off', gps: 'off', nMotion: 0, nOrient: 0, nGps: 0, acc: null, hz: 0 };
  let sendFn = null, metaSent = false, motionOn = false, gpsId = null, lastN = 0, lastT = performance.now();
  const r = (v, d = 4) => (v === null || v === undefined || Number.isNaN(v)) ? null : +v.toFixed(d);
  const push = (arr, row) => { if (arr.length < MAX) arr.push(row); };

  function onMotion(e) {
    const a = e.acceleration || {}, g = e.accelerationIncludingGravity || {}, w = e.rotationRate || {};
    push(B.motion, [Date.now(), r(a.x), r(a.y), r(a.z), r(g.x), r(g.y), r(g.z), r(w.alpha), r(w.beta), r(w.gamma)]);
    st.nMotion++; st.motion = 'on';
  }
  function onOrient(e) {
    const heading = e.webkitCompassHeading !== undefined ? e.webkitCompassHeading : (e.absolute ? (360 - e.alpha) % 360 : null);
    push(B.orient, [Date.now(), r(e.alpha, 2), r(e.beta, 2), r(e.gamma, 2), r(heading, 1)]);
    st.nOrient++;
  }
  function listen() {
    if (motionOn) return;
    motionOn = true;
    window.addEventListener('devicemotion', onMotion);
    window.addEventListener('deviceorientation', onOrient);
    window.addEventListener('deviceorientationabsolute', onOrient);
  }
  /* 사용자 터치 안에서 불러야 함 (아이폰) */
  async function askMotion() {
    if (motionOn) return;
    try {
      const D = window.DeviceMotionEvent, O = window.DeviceOrientationEvent;
      if (!D) { st.motion = 'none'; return; }
      if (typeof D.requestPermission === 'function') {
        const res = await D.requestPermission();
        if (O && typeof O.requestPermission === 'function') await O.requestPermission().catch(() => {});
        if (res !== 'granted') { st.motion = 'denied'; return; }
      }
      listen();
    } catch (e) { st.motion = 'waiting'; }
  }
  function startGps() {
    if (gpsId !== null) return;
    if (!navigator.geolocation) { st.gps = 'none'; return; }
    gpsId = navigator.geolocation.watchPosition(p => {
      const c = p.coords;
      push(B.gps, [p.timestamp || Date.now(), c.latitude, c.longitude, r(c.altitude, 1), r(c.accuracy, 1), r(c.altitudeAccuracy, 1), r(c.speed, 2), r(c.heading, 1)]);
      st.nGps++; st.gps = 'on'; st.acc = c.accuracy;
    }, e => { st.gps = e.code === 1 ? 'denied' : 'waiting'; }, { enableHighAccuracy: true, maximumAge: 0, timeout: 20000 });
  }
  function flush() {
    if (!sendFn) return;
    if (!B.motion.length && !B.orient.length && !B.gps.length) return;
    const m = { type: 'sensors', now: Date.now(), motion: B.motion.splice(0), orient: B.orient.splice(0), gps: B.gps.splice(0) };
    if (!metaSent) { m.meta = { ua: navigator.userAgent, screen: `${screen.width}x${screen.height}`, tz: Intl.DateTimeFormat().resolvedOptions().timeZone }; metaSent = true; }
    if (!sendFn(m)) { metaSent = metaSent && !m.meta; for (const k of ['motion', 'orient', 'gps']) B[k] = m[k].concat(B[k]).slice(-MAX); }
  }
  function badge() {
    let el = document.getElementById('sensLine');
    if (!el) {
      el = document.createElement('div'); el.id = 'sensLine'; el.setAttribute('aria-hidden', 'true');
      el.style.cssText = 'position:fixed;left:8px;bottom:6px;z-index:9;font:11px/1.3 system-ui;color:#fff;background:rgba(0,0,0,.55);padding:3px 7px;border-radius:8px;pointer-events:none';
      document.body.appendChild(el);
    }
    const now = performance.now(), dt = (now - lastT) / 1000;
    if (dt >= 1) { st.hz = Math.round((st.nMotion - lastN) / dt); lastN = st.nMotion; lastT = now; }
    const m = { on: `자이로 ${st.hz}Hz`, off: '자이로 대기', waiting: '자이로: 화면을 눌러 주세요', denied: '자이로 거부됨', none: '자이로 없음' }[st.motion];
    const g = { on: `GPS ±${Math.round(st.acc || 0)}m`, off: 'GPS 대기', waiting: 'GPS 찾는 중', denied: 'GPS 거부됨', none: 'GPS 없음' }[st.gps];
    el.textContent = `센서: ${m} · ${g}`;
  }
  function start(send) {
    sendFn = send;
    startGps();
    if (typeof (window.DeviceMotionEvent || {}).requestPermission === 'function') {
      st.motion = 'waiting';
      const once = () => { askMotion(); };
      document.addEventListener('pointerup', once, { once: true, capture: true });   // 첫 터치
    } else listen();
    setInterval(flush, 200);
    setInterval(badge, 500);
    badge();
  }
  return { start, askMotion, status: st };
})();
