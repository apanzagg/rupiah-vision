import { resolve } from 'node:path';

/** Opsi peluncuran Chrome dengan kamera palsu yang memutar berkas video (diulang terus). */
export function fakeCamera(video: string) {
  return {
    launchOptions: {
      args: [
        '--use-fake-ui-for-media-stream',
        '--use-fake-device-for-media-stream',
        `--use-file-for-fake-video-capture=${resolve(__dirname, '.fake', video)}`,
        '--autoplay-policy=no-user-gesture-required',
      ],
    },
  };
}

export interface DebugState {
  phase: string;
  announcements: string[];
  guidance: string[];
  frames: number;
  inferMs: number;
  fps: number;
}
