/* SayEye 상징 "빛 사람" — 얼굴 없이 은은하게 빛나는 사람 모양. 머리와 발밑은 따뜻한 빛, 몸은 푸른빛.
   mascot(el)   : el 안에 그린다
   el.classList : float(숨쉬듯 빛남) · look(안내 중, 빛이 천천히 흐름) · warn(전체가 주황빛으로) */
(function(){
  let n = 0;
  // 머리, 몸(어깨에서 아래로 퍼지는 종 모양), 발밑 빛
  const BODY = 'M100 98C142 98 162 126 164 158C165 180 170 192 168 198C166 202 160 204 152 204H48C40 204 34 202 32 198C30 192 35 180 36 158C38 126 58 98 100 98Z';
  window.mascot = function(el){
    const k = 'm' + (++n);
    el.classList.add('mascot');
    el.setAttribute('aria-hidden', 'true');
    el.innerHTML = `<svg viewBox="0 0 200 220">
  <defs>
    <linearGradient id="${k}b" x1="0" y1="0" x2="1" y2="0.3"><stop offset="0" stop-color="#8FD3E0"/><stop offset=".5" stop-color="#8C93D8"/><stop offset="1" stop-color="#C3A6D6"/></linearGradient>
    <radialGradient id="${k}h" cx="50%" cy="38%" r="60%"><stop offset="0" stop-color="#FFE7A8"/><stop offset=".55" stop-color="#FFB98A"/><stop offset="1" stop-color="#F49A86"/></radialGradient>
    <radialGradient id="${k}f" cx="50%" cy="50%" r="50%"><stop offset="0" stop-color="#FF9A4D"/><stop offset=".7" stop-color="#FF7A3D" stop-opacity=".85"/><stop offset="1" stop-color="#FF7A3D" stop-opacity="0"/></radialGradient>
    <linearGradient id="${k}w" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#FFC58F"/><stop offset=".6" stop-color="#FF8A5B"/><stop offset="1" stop-color="#F0624D"/></linearGradient>
    <filter id="${k}s" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="3.2"/></filter>
    <filter id="${k}g" x="-80%" y="-80%" width="260%" height="260%"><feGaussianBlur stdDeviation="14"/></filter>
  </defs>
  <g class="ms-aura" filter="url(#${k}g)">
    <circle cx="100" cy="62" r="40" fill="#FFC9A0" opacity=".55"/>
    <ellipse cx="100" cy="196" rx="70" ry="16" fill="#FF9A6A" opacity=".45"/>
    <ellipse cx="80" cy="150" rx="44" ry="50" fill="#9ED8E6" opacity=".35"/>
  </g>
  <g class="ms-body" filter="url(#${k}s)">
    <path d="${BODY}" fill="url(#${k}b)" opacity=".92"/>
    <path class="ms-warm" d="${BODY}" fill="url(#${k}w)"/>
    <ellipse cx="100" cy="196" rx="64" ry="15" fill="url(#${k}f)"/>
    <circle cx="100" cy="62" r="33" fill="url(#${k}h)"/>
    <ellipse cx="90" cy="48" rx="16" ry="9" fill="#FFF6D8" opacity=".55"/>
  </g>
</svg>`;
    return el;
  };
  document.addEventListener('DOMContentLoaded', () => document.querySelectorAll('[data-mascot]').forEach(window.mascot));
})();
