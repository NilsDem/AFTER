const CACHE_NAME = "after-midi-onnx-v4";

self.addEventListener("install", (event) => {
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((names) => Promise.all(names.filter((name) => !name.startsWith(CACHE_NAME)).map((name) => caches.delete(name))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  const cacheable =
    // url.pathname.startsWith("/AFTER/export_onnx/") ||
    url.pathname.startsWith("/export_onnx/") ||
    url.pathname.startsWith("../export_onnx/") ;
    // url.pathname.startsWith("/api/custom-models/");

  if (!cacheable || event.request.method !== "GET") {
    return;
  }

  event.respondWith(
    caches.open(CACHE_NAME).then(async (cache) => {
      if (event.request.cache === "reload" || event.request.cache === "no-store") {
        const response = await fetch(event.request);
        if (response.ok) {
          cache.put(event.request, response.clone());
        }
        return response;
      }

      const cached = await cache.match(event.request);
      if (cached) {
        return cached;
      }
      const response = await fetch(event.request);
      if (response.ok) {
        cache.put(event.request, response.clone());
      }
      return response;
    })
  );
});
