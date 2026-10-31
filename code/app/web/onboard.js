/* 처음 안내 화면: 그림 + 큰 제목 + 설명 + 점 + [다음] 버튼. 옆으로 밀어도 넘어간다.
   onboard({pages:[{art, title, text}], speak(text), done()}) — art 는 HTML 문자열(마스코트 자리는 <i data-mascot class="...">). */
window.onboard = function(opt){
  const pages = opt.pages, last = pages.length - 1;
  const root = document.createElement('section');
  root.className = 'onb'; root.setAttribute('role', 'dialog'); root.setAttribute('aria-modal', 'true'); root.setAttribute('aria-label', '처음 안내');
  root.innerHTML = `<div class="onb-top"><button class="onb-skip" type="button">건너뛰기</button></div>
    <div class="onb-track">${pages.map((p, i) => `<article class="onb-page" aria-hidden="${i ? 'true' : 'false'}">
      <div class="onb-art"><i class="halo"></i>${p.art}</div><h2>${p.title}</h2><p>${p.text}</p></article>`).join('')}</div>
    <div class="onb-dots" aria-hidden="true">${pages.map(() => '<i></i>').join('')}</div>
    <button class="onb-next" type="button">다음</button>`;
  document.body.append(root);
  root.querySelectorAll('[data-mascot]').forEach(el => window.mascot(el));
  const track = root.querySelector('.onb-track'), next = root.querySelector('.onb-next'),
        pg = [...root.querySelectorAll('.onb-page')], dots = [...root.querySelectorAll('.onb-dots i')];
  let cur = -1, spoken = false;
  function show(i, talk){
    cur = Math.max(0, Math.min(last, i));
    track.style.transform = `translateX(${-cur * 100}%)`;
    pg.forEach((p, k) => { p.classList.toggle('on', k === cur); p.setAttribute('aria-hidden', k !== cur); });
    dots.forEach((d, k) => d.classList.toggle('on', k === cur));
    next.textContent = cur === last ? '시작하기' : '다음';
    next.setAttribute('aria-label', `${cur + 1}/${pages.length}. ${pages[cur].title.replace(/\n/g, ' ')}. ${cur === last ? '시작하기' : '다음'}`);
    if(talk && opt.speak){ spoken = true; opt.speak(`${pages[cur].title}. ${pages[cur].text}`.replace(/\n/g, ' ')); }
  }
  function close(){
    root.classList.add('done'); setTimeout(() => root.remove(), 550);
    if(opt.done) opt.done();
  }
  next.onclick = () => { if(!spoken && opt.speak){ show(cur, true); if(cur < last) return; } cur === last ? close() : show(cur + 1, true); };
  root.querySelector('.onb-skip').onclick = close;
  pg.forEach(p => p.querySelector('.onb-art').addEventListener('click', () => show(cur, true)));   // 그림을 누르면 이 장을 읽어 준다
  let x0 = null;
  track.addEventListener('touchstart', e => { x0 = e.touches[0].clientX; }, {passive: true});
  track.addEventListener('touchend', e => { if(x0 === null) return; const d = e.changedTouches[0].clientX - x0; x0 = null;
    if(Math.abs(d) > 50) show(cur + (d < 0 ? 1 : -1), true); });
  root.addEventListener('keydown', e => { if(e.key === 'ArrowRight') show(cur + 1, true); if(e.key === 'ArrowLeft') show(cur - 1, true); });
  requestAnimationFrame(() => { show(0, false); next.focus({preventScroll: true}); });
  return {close};
};
