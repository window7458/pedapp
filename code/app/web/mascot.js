/* SayEye 마스코트 "세이" — 말랑한 유리 구름. 표정·색은 CSS 클래스로 바뀐다.
   mascot(el)        : el 안에 그린다
   el.classList      : look(두리번) · warn(따뜻한 경고색) · happy(웃음) · float(둥실) */
(function(){
  let n = 0;
  const BODY = 'M100 30C128 26 152 42 158 66C182 74 192 104 178 128C178 156 152 176 122 170C104 184 70 182 56 160C30 156 14 128 26 104C16 78 36 50 66 48C76 36 88 31 100 30Z';
  const INNER = 'M104 52C126 50 144 62 148 82C164 90 168 112 156 128C154 146 136 156 118 152C104 162 82 160 74 146C56 142 48 124 56 110C50 92 64 74 84 72C90 60 96 53 104 52Z';
  window.mascot = function(el){
    const k = 'm' + (++n);
    el.classList.add('mascot');
    el.setAttribute('aria-hidden', 'true');
    el.innerHTML = `<svg viewBox="0 0 200 210">
  <defs>
    <radialGradient id="${k}b" cx="38%" cy="30%" r="80%"><stop offset="0" stop-color="#E4ECFF"/><stop offset=".45" stop-color="#9DB2FA"/><stop offset="1" stop-color="#5D6CDA"/></radialGradient>
    <radialGradient id="${k}w" cx="38%" cy="30%" r="80%"><stop offset="0" stop-color="#FFF1DE"/><stop offset=".45" stop-color="#FFB27E"/><stop offset="1" stop-color="#F0624D"/></radialGradient>
    <linearGradient id="${k}g" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#fff" stop-opacity=".75"/><stop offset="1" stop-color="#fff" stop-opacity=".05"/></linearGradient>
    <filter id="${k}s" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="7"/></filter>
    <filter id="${k}h" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="4"/></filter>
  </defs>
  <ellipse class="ms-shadow" cx="102" cy="194" rx="58" ry="9" fill="#4A55B8" opacity=".22" filter="url(#${k}s)"/>
  <g class="ms-body">
    <path d="${BODY}" fill="url(#${k}b)"/>
    <path class="ms-warm" d="${BODY}" fill="url(#${k}w)"/>
    <path d="${INNER}" fill="url(#${k}g)" opacity=".55"/>
    <ellipse cx="74" cy="66" rx="20" ry="10" fill="#fff" opacity=".7" filter="url(#${k}h)" transform="rotate(-24 74 66)"/>
    <g class="ms-face">
      <g class="ms-eyes"><ellipse class="ms-eye" cx="84" cy="104" rx="7.5" ry="10" fill="#1D2250"/><ellipse class="ms-eye" cx="122" cy="104" rx="7.5" ry="10" fill="#1D2250"/>
        <circle cx="86.5" cy="100" r="2.4" fill="#fff"/><circle cx="124.5" cy="100" r="2.4" fill="#fff"/></g>
      <path class="ms-mouth" d="M90 126q13 11 26 0" fill="none" stroke="#1D2250" stroke-width="5" stroke-linecap="round"/>
      <ellipse cx="68" cy="122" rx="9" ry="5" fill="#FF9E8A" opacity=".35"/><ellipse cx="138" cy="122" rx="9" ry="5" fill="#FF9E8A" opacity=".35"/>
    </g>
  </g>
</svg>`;
    return el;
  };
  document.addEventListener('DOMContentLoaded', () => document.querySelectorAll('[data-mascot]').forEach(window.mascot));
})();
