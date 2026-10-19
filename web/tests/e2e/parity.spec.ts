// Parity test: output onnxruntime-web (pipeline web lengkap: letterbox canvas -> worker -> decode -> NMS)
// harus cocok dengan referensi Python (training/scripts/infer_ref.py) pada citra uji yang sama.
import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import modelCfg from '../../src/model-config.json';

interface Det {
  cls: number;
  score: number;
  box: number[];
}
const ref = JSON.parse(readFileSync(resolve(import.meta.dirname, '../../public/parity/parity_ref.json'), 'utf8')) as {
  images: Record<string, Record<string, Det[]>>;
};
const MIN_SCORE = 0.3; // abaikan deteksi di sekitar ambang 0,25 yang bisa muncul/hilang karena beda resize
const iou = (a: number[], b: number[]) => {
  const iw = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0]));
  const ih = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
  const inter = iw * ih;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
};

test('parity web vs Python', async ({ page }) => {
  await page.goto(`./benchmark/?parity=1&model=${modelCfg.file}`);
  const res = (await page.waitForFunction(() => (window as unknown as { __parity?: unknown }).__parity, null, { timeout: 120_000 }).then((h) =>
    h.jsonValue(),
  )) as { result: Record<string, Det[]> };
  const report: unknown[] = [];
  for (const [name, refByVariant] of Object.entries(ref.images)) {
    const expected = (refByVariant[modelCfg.variant] ?? refByVariant.fp32).filter((d) => d.score >= MIN_SCORE);
    const got = res.result[name].filter((d) => d.score >= MIN_SCORE);
    for (const e of expected) {
      const m = got.find((g) => g.cls === e.cls && iou(g.box, e.box) >= 0.9);
      report.push({ name, cls: e.cls, ref: e.score, web: m?.score, iou: m ? iou(m.box, e.box) : 0 });
      expect(m, `${name}: kelas ${e.cls} (skor ${e.score}) tidak ditemukan di web`).toBeTruthy();
      expect(Math.abs(m!.score - e.score), `${name}: selisih skor`).toBeLessThanOrEqual(0.05);
    }
    for (const g of got) {
      expect(expected.some((e) => e.cls === g.cls && iou(g.box, e.box) >= 0.9), `${name}: deteksi web tambahan kelas ${g.cls}`).toBe(true);
    }
  }
  test.info().annotations.push({ type: 'parity', description: JSON.stringify(report) });
});
