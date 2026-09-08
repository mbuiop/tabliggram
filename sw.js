// دخل و خرج — service worker برای کارکرد آفلاین
// این فایل باید کنار dakhl-o-kharj.html و manifest.json، روی یک وب‌سرور واقعی
// (http:// یا https://) سرو بشه؛ باز کردن مستقیم فایل (file:// یا content://)
// باعث می‌شه ثبت service worker رد بشه — این محدودیت خود مرورگرهاست، نه این کد.

const CACHE_NAME = 'لاتاری';
const APP_SHELL = [
  './index.html,
  './manifest.json',
  './icon-192.png',
  './icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// cache-first for the app shell, network-first (with cache fallback) for everything else
// (so live prices/job-search style network calls still try the network first)
self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;

  const isAppShell = APP_SHELL.some((path) => req.url.endsWith(path.replace('./', '')));

  if (isAppShell) {
    event.respondWith(
      caches.match(req).then((cached) => cached || fetch(req))
    );
    return;
  }

  event.respondWith(
    fetch(req)
      .then((res) => {
        const resClone = res.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(req, resClone)).catch(() => {});
        return res;
      })
      .catch(() => caches.match(req))
  );
});
