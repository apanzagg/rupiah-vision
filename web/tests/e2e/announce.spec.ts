// E2E: aplikasi berjalan dengan kamera palsu (video uang) dan mengumumkan nominal yang benar.
import { chromium, expect, test, type Page } from '@playwright/test';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fakeCamera, type DebugState } from './fake-camera';

// dibuat oleh make_fake_videos.py (tidak di-commit karena berukuran besar)
const casesFile = resolve(import.meta.dirname, '.fake', 'cases.json');
const cases: { video: string; expect: string; kind: string }[] = existsSync(casesFile) ? JSON.parse(readFileSync(casesFile, 'utf8')) : [];

const dbg = (page: Page) => page.evaluate(() => (window as unknown as { __suararupiah: DebugState }).__suararupiah);

// Setiap kasus butuh video kamera palsu sendiri (argumen peluncuran Chrome). Playwright >= 1.5x tidak mengizinkan
// test.use({ launchOptions }) di dalam describe, jadi browser diluncurkan langsung per kasus.
for (const c of cases) {
  test(`${c.kind}: ${c.video} -> ${c.expect}`, async ({ baseURL, channel, headless, viewport }) => {
    const browser = await chromium.launch({ channel, headless, ...fakeCamera(c.video).launchOptions });
    const context = await browser.newContext({ baseURL, viewport, permissions: ['camera'] });
    const page = await context.newPage();
    try {
      const t0 = Date.now();
      await page.goto('./');
      await expect.poll(async () => (await dbg(page))?.phase, { timeout: 90_000 }).toBe('ready');
      const readyMs = Date.now() - t0;
      if (c.kind === 'nominal') {
        await expect.poll(async () => (await dbg(page)).announcements.join(' | '), { timeout: 60_000 }).toContain(c.expect);
        // region aria-live berisi pengumuman yang sama
        await expect(page.locator('#live')).toContainText(c.expect, { timeout: 5_000 });
        const s = await dbg(page);
        expect(s.announcements.every((a) => a.includes(c.expect))).toBe(true); // tidak ada nominal lain yang disebut
      } else {
        await expect.poll(async () => (await dbg(page)).guidance.join(' | '), { timeout: 60_000 }).toContain(c.expect);
        expect((await dbg(page)).announcements).toEqual([]); // tidak boleh menyebut nominal
      }
      const s = await dbg(page);
      test.info().annotations.push({ type: 'perf', description: JSON.stringify({ readyMs, frames: s.frames, inferMs: s.inferMs }) });
    } finally {
      await browser.close();
    }
  });
}

// Offline: setelah kunjungan pertama, halaman dimuat ulang TANPA jaringan dan tetap mengumumkan nominal
// (aplikasi, model, dan WASM disajikan dari cache service worker / Cache Storage).
const offlineCase = cases.find((c) => c.video === 'rp100000.y4m') ?? cases.find((c) => c.kind === 'nominal');
if (offlineCase) {
  test(`offline: ${offlineCase.video} -> ${offlineCase.expect}`, async ({ baseURL, channel, headless, viewport }) => {
    const browser = await chromium.launch({ channel, headless, ...fakeCamera(offlineCase.video).launchOptions });
    const context = await browser.newContext({ baseURL, viewport, permissions: ['camera'] });
    const page = await context.newPage();
    try {
      await page.goto('./');
      await expect.poll(async () => (await dbg(page))?.phase, { timeout: 90_000 }).toBe('ready');
      await page.waitForFunction(() => navigator.serviceWorker?.ready.then(() => true), null, { timeout: 30_000 });
      await context.setOffline(true);
      await page.reload();
      await expect.poll(async () => (await dbg(page))?.phase, { timeout: 60_000 }).toBe('ready');
      expect(await page.evaluate(() => navigator.onLine)).toBe(false);
      await expect.poll(async () => (await dbg(page)).announcements.join(' | '), { timeout: 60_000 }).toContain(offlineCase.expect);
    } finally {
      await browser.close();
    }
  });
}
