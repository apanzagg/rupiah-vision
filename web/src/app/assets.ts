// Lokasi & versi aset besar. Mengganti model = perbarui src/model-config.json (versi baru -> cache baru).
import wasmAssetUrl from 'onnxruntime-web/ort-wasm-simd-threaded.wasm?url';
import modelCfg from '../model-config.json';
import manifest from '../../public/audio/manifest.json';

export const BASE = import.meta.env.BASE_URL;
export const MODEL = modelCfg;
export const MODEL_URL = `${BASE}models/${modelCfg.file}`;
export const ORT_VERSION = '1.30.0';
/** WASM bisa dikirim terkompresi gzip (lihat vite.config.ts: WASM_GZ) jika server tidak mengompresnya. */
export const WASM_URL = import.meta.env.VITE_WASM_GZ ? `${BASE}ort/ort-wasm-simd-threaded.wasm.gz` : wasmAssetUrl;
export const WASM_BYTES = 14_000_000;
export const CACHE_PREFIX = 'suararupiah-asset-';
export const MODEL_CACHE = `${CACHE_PREFIX}model-${modelCfg.version}-${modelCfg.variant}`;
export const ORT_CACHE = `${CACHE_PREFIX}ort-${ORT_VERSION}`;
export const AUDIO_MANIFEST = manifest;
export const SOURCE_URL = 'https://github.com/apanzagg/rupiah-vision';
