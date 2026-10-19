// PWA: dapat dipasang (installable), berfungsi offline setelah kunjungan pertama, dan waktu muat pertama vs dari cache.
// Lighthouse >= 12 tidak lagi punya kategori PWA, jadi kelayakan instal diperiksa langsung lewat Chrome DevTools
// Protocol (Page.getInstallabilityErrors) dan service worker.
import { expect, test, type Page } from '@playwright/test';
import { mkdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fakeCamera } from './fake-camera';

test.use(fakeCamera('kosong.y4m'));

const phase = (page: Page) => page.evaluate(() => (window as unknown as { __suararupiah?: { phase: string } }).__suararupiah?.phase);

test('installable, offline, dan waktu muat', async ({ page, context }) => {
  const t0 = Date.now();
  await page.goto('./');
  await expect.poll(() => phase(page), { timeout: 120_000 }).toBe('ready');
  const firstLoadMs = Date.now() - t0;

  // service worker aktif & mengendalikan halaman setelah reload
  await page.waitForFunction(() => navigator.serviceWorker?.ready.then(() => true), null, { timeout: 30_000 });
  const t1 = Date.now();
  await page.reload();
  await expect.poll(() => phase(page), { timeout: 60_000 }).toBe('ready');
  const cachedLoadMs = Date.now() - t1;
  expect(await page.evaluate(() => !!navigator.serviceWorker.controller)).toBe(true);

  // kelayakan instal (kriteria Chrome): tidak ada error
  const cdp = await context.newCDPSession(page);
  const { installabilityErrors } = (await cdp.send('Page.getInstallabilityErrors')) as {
    installabilityErrors: { errorId: string }[];
  };
  // konteks Playwright setara mode penyamaran ("in-incognito"): itu kondisi uji, bukan kekurangan aplikasi
  expect(installabilityErrors.filter((e) => e.errorId !== 'in-incognito')).toEqual([]);
  const manifest = (await cdp.send('Page.getAppManifest')) as { url: string; errors: unknown[] };
  expect(manifest.errors).toEqual([]);

  // offline: muat ulang tanpa jaringan -> tetap siap (model & WASM dari Cache Storage, aset dari service worker)
  await context.setOffline(true);
  const t2 = Date.now();
  await page.reload();
  await expect.poll(() => phase(page), { timeout: 60_000 }).toBe('ready');
  const offlineLoadMs = Date.now() - t2;
  await context.setOffline(false);

  const res = { firstLoadMs, cachedLoadMs, offlineLoadMs, installable: true, manifest: manifest.url };
  mkdirSync(resolve(import.meta.dirname, '../../test-results'), { recursive: true });
  writeFileSync(resolve(import.meta.dirname, '../../test-results/pwa_load_times.json'), JSON.stringify(res, null, 1));
  test.info().annotations.push({ type: 'pwa', description: JSON.stringify(res) });
});
