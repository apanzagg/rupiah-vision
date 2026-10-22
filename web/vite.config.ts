import { resolve } from 'node:path';
import { defineConfig } from 'vite';
import { VitePWA } from 'vite-plugin-pwa';

// BASE_PATH: '/rupiah-vision/' untuk GitHub Pages, '/' untuk lokal.
const base = process.env.BASE_PATH ?? '/';

export default defineConfig({
  base,
  build: {
    target: 'es2022',
    rollupOptions: {
      input: {
        main: resolve(import.meta.dirname, 'index.html'),
        about: resolve(import.meta.dirname, 'about/index.html'),
        benchmark: resolve(import.meta.dirname, 'benchmark/index.html'),
      },
    },
  },
  worker: { format: 'es' },
  plugins: [
    VitePWA({
      registerType: 'autoUpdate',
      injectRegister: 'script-defer',
      includeAssets: ['icons/*.png'],
      manifest: {
        name: 'SuaraRupiah',
        short_name: 'SuaraRupiah',
        description: 'Pengenal nominal uang kertas Rupiah dengan suara, untuk penyandang tunanetra. Berjalan di ponsel tanpa mengirim gambar.',
        lang: 'id',
        dir: 'ltr',
        start_url: './',
        scope: './',
        display: 'standalone',
        orientation: 'portrait',
        background_color: '#000000',
        theme_color: '#000000',
        categories: ['accessibility', 'utilities'],
        icons: [
          { src: 'icons/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: 'icons/icon-512.png', sizes: '512x512', type: 'image/png' },
          { src: 'icons/icon-maskable-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        // Model ONNX & runtime WASM TIDAK di-precache service worker: aplikasi menyimpannya sendiri di
        // Cache Storage (src/app/loader.ts) agar progres unduhan bisa ditampilkan dan tidak terunduh dua kali.
        globPatterns: ['**/*.{js,css,html,png,svg,json,mp3,webmanifest,woff2}'],
        globIgnores: ['**/models/**', '**/parity/**', '**/bench/**', '**/*.wasm', '**/*.gz'],
        ignoreURLParametersMatching: [/.*/],
        cleanupOutdatedCaches: true,
        clientsClaim: true,
        skipWaiting: true,
        maximumFileSizeToCacheInBytes: 3 * 1024 * 1024,
      },
    }),
  ],
});
