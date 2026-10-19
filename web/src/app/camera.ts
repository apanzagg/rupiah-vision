// Kamera belakang + senter (torch) bila didukung.

export interface CameraHandle {
  stream: MediaStream;
  track: MediaStreamTrack;
  hasTorch: boolean;
}

export class CameraError extends Error {
  constructor(public kind: 'denied' | 'unavailable', message: string) {
    super(message);
  }
}

export async function openCamera(video: HTMLVideoElement): Promise<CameraHandle> {
  if (!navigator.mediaDevices?.getUserMedia) throw new CameraError('unavailable', 'getUserMedia tidak didukung');
  let stream: MediaStream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } },
    });
  } catch (e) {
    const name = (e as DOMException)?.name;
    throw new CameraError(name === 'NotAllowedError' || name === 'SecurityError' ? 'denied' : 'unavailable', String(e));
  }
  video.srcObject = stream;
  video.muted = true;
  video.playsInline = true;
  await video.play().catch(() => undefined);
  if (!video.videoWidth) await new Promise((r) => video.addEventListener('loadedmetadata', r, { once: true }));
  const track = stream.getVideoTracks()[0];
  const caps = (track.getCapabilities?.() ?? {}) as MediaTrackCapabilities & { torch?: boolean };
  return { stream, track, hasTorch: !!caps.torch };
}

export async function setTorch(cam: CameraHandle, on: boolean): Promise<boolean> {
  if (!cam.hasTorch) return false;
  try {
    await cam.track.applyConstraints({ advanced: [{ torch: on } as MediaTrackConstraintSet] });
    return true;
  } catch {
    return false;
  }
}

export function stopCamera(cam: CameraHandle | null): void {
  cam?.stream.getTracks().forEach((t) => t.stop());
}
