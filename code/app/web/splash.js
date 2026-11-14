/* SayEye 로딩 화면 — 앱이 하는 일을 3초 안에 보여 준다.
   ① 폰을 든 빛나는 사람이 나타남 → ② 폰 카메라가 앞을 비춤(빛줄기) → ③ 앞에 다가오는 차가 잡힘(테두리)
   → ④ 폰에서 소리 물결(음성 안내) → ⑤ "SayEye" 이름. 자막: 앞을 대신 보고 · 위험을 먼저 말하고 · 보호자와 함께 걸어요 */
(function () {
  const caps = window.SPLASH_CAPS || ["앞을 대신 보고", "위험을 먼저 말하고", "보호자와 함께 걸어요"];
  const svg = `<svg class="fig" viewBox="0 0 390 660" preserveAspectRatio="xMidYMid meet" aria-hidden="true">
  <defs>
    <filter id="spS" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="9"/></filter>
    <filter id="spS2" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="22"/></filter>
    <filter id="spP" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2.5"/></filter>
    <filter id="spB" x="-20%" y="-50%" width="140%" height="200%"><feGaussianBlur stdDeviation="10"/></filter>
    <linearGradient id="spFg" x1="0" y1="0" x2="0" y2="1"><stop offset=".55" stop-color="#fff"/><stop offset=".92" stop-color="#000"/></linearGradient>
    <mask id="spFade" maskUnits="userSpaceOnUse" x="0" y="0" width="390" height="660"><rect width="390" height="660" fill="url(#spFg)"/></mask>
    <mask id="spSil" maskUnits="userSpaceOnUse" x="0" y="0" width="390" height="660"><g filter="url(#spS)" fill="#fff">
      <ellipse cx="208" cy="122" rx="50" ry="60" transform="rotate(12 208 122)"/>
      <path d="M176 176C150 190 126 216 118 262C108 332 102 450 92 660L330 660C322 450 316 330 302 256C294 214 264 188 234 178Z"/>
      <path d="M296 296C286 362 232 390 158 362" stroke="#fff" stroke-width="48" stroke-linecap="round" fill="none"/>
      <circle cx="150" cy="352" r="28"/></g></mask>
    <linearGradient id="spBeam" x1="1" y1="0" x2="0" y2="0"><stop offset="0" stop-color="#fff" stop-opacity=".75"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></linearGradient>
  </defs>
  <g class="walker">
    <g filter="url(#spS2)" opacity=".5" mask="url(#spFade)"><ellipse cx="210" cy="320" rx="130" ry="260" fill="#7FB8FF"/></g>
    <g mask="url(#spFade)"><g mask="url(#spSil)"><g filter="url(#spS2)">
      <rect width="390" height="660" fill="#4E7FF2"/>
      <circle cx="212" cy="82" r="72" fill="#E6F6FF"/><circle cx="285" cy="240" r="66" fill="#8FDCFF"/>
      <circle cx="200" cy="168" r="38" fill="#C2EEFF"/><circle cx="150" cy="250" r="70" fill="#9FB4FF"/>
      <circle cx="210" cy="500" r="120" fill="#2F5CD6"/><circle cx="160" cy="400" r="80" fill="#5FA8FF"/>
      <circle cx="150" cy="352" r="62" fill="#7CD0FF"/>
    </g></g></g>
  </g>
  <path class="beam" d="M120 312 L-10 228 L-10 420 Z" fill="url(#spBeam)" filter="url(#spB)"/>
  <g class="hit"><rect x="26" y="300" width="58" height="40" rx="10" fill="#A9E4FF" opacity=".6" filter="url(#spP)"/>
    <rect x="18" y="292" width="74" height="56" rx="12" fill="none" stroke="#fff" stroke-width="3" stroke-dasharray="14 8"/></g>
  <circle class="ring" cx="138" cy="318" r="18" fill="none" stroke="#D8F3FF" stroke-width="2.5"/>
  <circle class="ring r2" cx="138" cy="318" r="18" fill="none" stroke="#D8F3FF" stroke-width="2.5"/>
  <rect x="122" y="290" width="32" height="56" rx="7" fill="#F2FAFF" transform="rotate(-22 138 318)" filter="url(#spP)"/>
</svg>`;
  const el = document.createElement("div");
  el.id = "splash";
  el.setAttribute("aria-hidden", "true");
  el.innerHTML = svg + `<div class="cap">${caps.map((c, i) => `<span style="animation-delay:${0.7 + i * 0.85}s;animation-duration:.95s">${c}</span>`).join("")}</div>
    <div class="name">${[..."SayEye"].map((c, i) => `<span style="animation-delay:${2.9 + i * 0.07}s">${c}</span>`).join("")}</div>`;
  document.body.prepend(el);
  const reduce = matchMedia("(prefers-reduced-motion:reduce)").matches, t0 = performance.now();
  const hide = () => { el.classList.add("done"); setTimeout(() => el.remove(), 800); window.dispatchEvent(new Event("splashdone")); };
  addEventListener("load", () => setTimeout(hide, Math.max(0, (reduce ? 300 : 4600) - (performance.now() - t0))));
  el.addEventListener("click", hide, { once: true });       // 눌러서 건너뛰기
})();
