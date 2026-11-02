/* SayEye 보호자 푸시 알림 서비스 워커 — 보호자 화면을 닫아도 위험 알림이 폰 알림으로 뜬다. */
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("push", (e) => {
  let d = {};
  try { d = e.data ? e.data.json() : {}; } catch (_) { d = { title: "SayEye", body: e.data && e.data.text() }; }
  const strong = ["urgent", "sos", "offline"].includes(d.level);
  e.waitUntil(self.registration.showNotification(d.title || "SayEye 보호자", {
    body: d.body || "새 알림이 있어요.",
    tag: d.tag || "sayeye",
    renotify: true,
    requireInteraction: strong,
    vibrate: strong ? [300, 120, 300, 120, 300] : [160],
    icon: "/icon-192.png",
    badge: "/icon-192.png",
    data: { url: d.url || "/guardian", id: d.id },
  }));
});

self.addEventListener("notificationclick", (e) => {
  e.notification.close();
  const url = (e.notification.data && e.notification.data.url) || "/guardian";
  e.waitUntil((async () => {
    const all = await self.clients.matchAll({ type: "window", includeUncontrolled: true });
    for (const c of all) { if (c.url.includes("/guardian")) { await c.focus(); return; } }
    await self.clients.openWindow(url);
  })());
});
