import { defineConfig } from '@playwright/test';

// E2E memakai Google Chrome yang terpasang (channel 'chrome') dengan kamera palsu dari berkas video .y4m
// (dibuat oleh tests/e2e/make_fake_videos.py). Setiap berkas spec memilih videonya sendiri lewat test.use().
export default defineConfig({
  testDir: 'tests/e2e',
  timeout: 180_000,
  workers: 1,
  reporter: [['list'], ['json', { outputFile: 'test-results/e2e-results.json' }]],
  use: {
    channel: process.env.PW_CHANNEL ?? 'chrome',
    baseURL: 'http://localhost:4173/',
    permissions: ['camera'],
    viewport: { width: 412, height: 860 },
  },
  webServer: {
    command: 'npx vite preview --port 4173 --strictPort',
    url: 'http://localhost:4173/',
    reuseExistingServer: true,
    timeout: 60_000,
  },
});
