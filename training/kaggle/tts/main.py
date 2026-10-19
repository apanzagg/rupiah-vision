# Kernel Kaggle "suararupiah-tts": buat klip audio semua frasa dengan dua mesin TTS lalu ukur
# keterbacaannya dengan ASR (Whisper large-v3-turbo). Mesin dipilih berdasarkan CER + lisensi.
#   mms   : facebook/mms-tts-ind (VITS), lisensi CC-BY-NC 4.0
#   piper : rhasspy/piper-voices id_ID-news_tts-medium (Piper, kode MIT)
# Output: audio_mms/*.mp3, audio_piper/*.mp3, asr_report.json
import json
import re
import subprocess
import sys
import time
import urllib.request
import wave
from pathlib import Path

T0 = time.time()
W = Path("/kaggle/working")
TMP = Path("/tmp/tts")
TMP.mkdir(parents=True, exist_ok=True)


def sh(cmd: str) -> None:
    print(f"\n$ {cmd}   [{time.time() - T0:.0f} dtk]", flush=True)
    subprocess.run(cmd, shell=True, check=True)


sys.path.insert(0, "/tmp/src/scripts")
from tts_phrases import num_to_words  # noqa: E402

manifest = json.load(open("/tmp/src/manifest.json", encoding="utf-8"))
PHR = {k: v["text"] for k, v in manifest["phrases"].items()}
print(len(PHR), "frasa")


def to_mp3(wav: Path, mp3: Path) -> None:
    flt = ("silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
           "silenceremove=start_periods=1:start_threshold=-45dB,areverse,"
           "adelay=60|60,apad=pad_dur=0.08,loudnorm=I=-16:TP=-1.5:LRA=11")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav), "-af", flt, "-ac", "1", "-ar", "22050",
                    "-b:a", "40k", str(mp3)], check=True)


# ---------------- MMS-TTS ----------------
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.io import wavfile  # noqa: E402
from transformers import AutoTokenizer, VitsModel  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained("facebook/mms-tts-ind")
mms = VitsModel.from_pretrained("facebook/mms-tts-ind").to(dev).eval()
out_mms = W / "audio_mms"
out_mms.mkdir(exist_ok=True)
for k, text in PHR.items():
    torch.manual_seed(0)
    with torch.no_grad():
        wavf = mms(**tok(text, return_tensors="pt").to(dev)).waveform[0].cpu().numpy()
    p = TMP / f"mms_{k}.wav"
    wavfile.write(p, mms.config.sampling_rate, (np.clip(wavf, -1, 1) * 32767).astype(np.int16))
    to_mp3(p, out_mms / f"{k}.mp3")
print(f"MMS selesai [{time.time() - T0:.0f} dtk]", flush=True)
del mms

# ---------------- Piper ----------------
sh("pip install -q piper-tts")
base = "https://huggingface.co/rhasspy/piper-voices/resolve/main/id/id_ID/news_tts/medium/"
for f in ("id_ID-news_tts-medium.onnx", "id_ID-news_tts-medium.onnx.json", "MODEL_CARD"):
    urllib.request.urlretrieve(base + f, TMP / f)
from piper import PiperVoice  # noqa: E402

voice = PiperVoice.load(str(TMP / "id_ID-news_tts-medium.onnx"))
out_piper = W / "audio_piper"
out_piper.mkdir(exist_ok=True)
for k, text in PHR.items():
    p = TMP / f"piper_{k}.wav"
    with wave.open(str(p), "wb") as wf:
        if hasattr(voice, "synthesize_wav"):
            voice.synthesize_wav(text, wf)
        else:
            voice.synthesize(text, wf)
    to_mp3(p, out_piper / f"{k}.mp3")
print(f"Piper selesai [{time.time() - T0:.0f} dtk]", flush=True)

# ---------------- ASR (Whisper) ----------------
from transformers import pipeline  # noqa: E402

asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo",
               torch_dtype=torch.float16 if dev == "cuda" else torch.float32, device=0 if dev == "cuda" else -1)


def norm(s: str) -> str:
    s = s.lower()
    s = re.sub(r"rp\.?\s*(\d[\d.,]*)", lambda m: m.group(1) + " rupiah", s)
    def words(m):  # angka -> kata; angka mustahil (halusinasi ASR) dibiarkan sebagai digit
        v = int(re.sub(r"[.,]", "", m.group(0)) or 0)
        return " " + (num_to_words(v) if v < 1_000_000_000 else m.group(0)) + " "

    s = re.sub(r"\d[\d.,]*", words, s)
    s = re.sub(r"[^a-z ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def cer(ref: str, hyp: str) -> float:
    a, b = ref.replace(" ", ""), hyp.replace(" ", "")
    d = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, d[0] = d[0], i
        for j, cb in enumerate(b, 1):
            cur = min(d[j] + 1, d[j - 1] + 1, prev + (ca != cb))
            prev, d[j] = d[j], cur
    return d[len(b)] / max(1, len(a))


report = {}
for eng, folder in (("mms", out_mms), ("piper", out_piper)):
    rows = {}
    for k, text in PHR.items():
        hyp = asr(str(folder / f"{k}.mp3"), return_timestamps=True,
                  generate_kwargs={"language": "indonesian", "task": "transcribe"})["text"]
        rows[k] = {"ref": norm(text), "hyp": norm(hyp), "raw": hyp.strip(), "cer": round(cer(norm(text), norm(hyp)), 4)}
    cers = [r["cer"] for r in rows.values()]
    size = sum(p.stat().st_size for p in folder.glob("*.mp3"))
    report[eng] = {"mean_cer": round(float(np.mean(cers)), 4), "exact_match": sum(c == 0 for c in cers),
                   "n": len(cers), "total_kb": round(size / 1024), "clips": rows}
    print(eng, {k: v for k, v in report[eng].items() if k != "clips"}, flush=True)
(W / "asr_report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
sh("cp /tmp/tts/MODEL_CARD /kaggle/working/piper_MODEL_CARD.txt")
print(f"SELESAI dalam {time.time() - T0:.0f} dtk")
