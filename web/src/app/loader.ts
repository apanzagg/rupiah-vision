// Unduh aset besar (model ONNX, runtime WASM) dengan progres, simpan di Cache Storage agar berfungsi offline.
// Nama cache memuat versi, sehingga mengganti model = mengganti versi -> cache lama dibersihkan.

export type Progress = (received: number, total: number) => void;

async function readWithProgress(res: Response, expected: number, onProgress?: Progress): Promise<ArrayBuffer> {
  const len = Number(res.headers.get('content-length')) || 0;
  const encoded = !!res.headers.get('content-encoding');
  const total = !encoded && len ? len : expected;
  if (!res.body) {
    const buf = await res.arrayBuffer();
    onProgress?.(buf.byteLength, buf.byteLength);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress?.(received, Math.max(total, received));
  }
  const out = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    out.set(c, off);
    off += c.byteLength;
  }
  return out.buffer;
}

/** Ambil dari Cache Storage bila ada; jika belum, unduh (dengan progres) lalu simpan. */
export async function fetchCached(url: string, cacheName: string, expectedBytes: number, onProgress?: Progress): Promise<ArrayBuffer> {
  let cache: Cache | null = null;
  try {
    cache = await caches.open(cacheName);
    const hit = await cache.match(url);
    if (hit) {
      const buf = await hit.arrayBuffer();
      onProgress?.(buf.byteLength, buf.byteLength);
      return buf;
    }
  } catch {
    cache = null; // Cache Storage tidak tersedia (mis. mode privat): tetap jalan tanpa cache
  }
  const res = await fetch(url, { cache: 'no-cache' });
  if (!res.ok) throw new Error(`Gagal mengunduh ${url}: HTTP ${res.status}`);
  let buf = await readWithProgress(res, expectedBytes, onProgress);
  if (url.endsWith('.gz')) {
    // berkas dikompres manual (GitHub Pages tidak selalu mengompres .wasm)
    const ds = new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip')));
    buf = await ds.arrayBuffer();
  }
  try {
    await cache?.put(url, new Response(buf.slice(0), { headers: { 'Content-Type': 'application/octet-stream' } }));
  } catch {
    /* kuota penuh: abaikan */
  }
  return buf;
}

/** Hapus cache aset versi lama. */
export async function cleanupCaches(prefix: string, keep: string[]): Promise<void> {
  try {
    for (const name of await caches.keys()) {
      if (name.startsWith(prefix) && !keep.includes(name)) await caches.delete(name);
    }
  } catch {
    /* abaikan */
  }
}
