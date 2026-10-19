// E2E: aplikasi berjalan dengan kamera palsu (video uang) dan mengumumkan nominal yang benar.
import { expect, test, type Page } from '@playwright/test';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fakeCamera, type DebugState } from './fake-camera';

// dibuat oleh make_fake_videos.py (tidak di-commit karena berukuran besar)
const casesFile = resolve(__dirname, '.fake', 'cases.json');
const cases: { video: string; expect: string; kind: string }[] = existsSync(casesFile) ? JSON.parse(readFileSync(casesFile, 'utf8')) : [];

const dbg = (page: Page) => page.evaluate(() => (window as unknown as { __suararupiah: DebugState }).__suararupiah);

for (const c of cases) {
  test.describe(`${c.kind}: ${c.video}`, () => {
    test.use(fakeCamera(c.video));

    test(`mengumumkan/menuntun dengan benar (${c.expect})`, async ({ page }) => {
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
    });
  });
}
