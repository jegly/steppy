"""
Steppy — ACE-Step 1.5 inference worker.

Drop-in sibling of the stable-audio-3 worker.py: same resident-process design,
same newline-delimited JSON protocol over stdin/stdout, so the existing GTK GUI
can drive it by pointing SF_WORKER at this file instead. worker.py is NOT
modified and NOT imported.

Protocol (stdin commands) — superset of worker.py's:
  {"cmd": "generate", "id": 1, "prompt": "...", "duration": 30, "steps": 8,
   "cfg": 1.0, "seed": -1, "spectrogram": true, "out": "/path/out.wav",
   --- lyrics-engine additions ---
   "lyrics": "[Verse 1]\\n...",      # explicit lyrics; wins over auto_lyrics
   "auto_lyrics": false,             # true -> 5Hz LM writes them from "prompt"
   "instrumental": false,            # true -> force no vocals
   "bpm": null, "keyscale": "", "timesignature": "", "language": "en"}
  {"cmd": "ping"}

Events (stdout) — identical to worker.py:
  {"event":"loading"} | {"event":"ready"} | {"event":"error","msg":...}
  {"event":"progress","id":1,"step":i,"total":n}
  {"event":"stage","id":1,"name":"..."}
  {"event":"done","id":1,"path":"...","spectrogram":"...","seconds":30.0,"seed":123}
  plus {"lyrics": "..."} on done when the LM wrote them.

Environment:
  SF_STEPPY_ACESTEP_ROOT   ACE-Step-1.5 project root (contains the acestep package)
  SF_STEPPY_ACESTEP_CKPT   checkpoints dir (default $SF_STEPPY_ACESTEP_ROOT/checkpoints)
  SF_STEPPY_ACESTEP_DIT    DiT config name (default acestep-v15-turbo)
  SF_STEPPY_ACESTEP_LM     LM model dir name (default acestep-5Hz-lm-0.6B)
  SF_LOWMEM=1       int8-quantise the DiT and load bf16 (the 16GB build)

Hardware note: pre-AVX2 CPUs (e.g. Ivy Bridge) crash with SIGILL inside oneDNN
on some tensor shapes — a 30s render succeeds while a 60s one dies. We clamp
DNNL_MAX_CPU_ISA to AVX when the host lacks avx2. Verified to cost nothing:
14.6x realtime unclamped vs 14.9x clamped.
"""
import json
import os
import sys
import time
import wave

import numpy as np


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def _cpu_has(flag):
    try:
        with open("/proc/cpuinfo", "r") as fh:
            for line in fh:
                if line.startswith("flags"):
                    return f" {flag} " in f" {line.split(':',1)[1].strip()} "
    except OSError:
        pass
    return False


# --- must happen BEFORE torch is imported ---
if not _cpu_has("avx2"):
    os.environ.setdefault("DNNL_MAX_CPU_ISA", "AVX")
LOWMEM = os.environ.get("SF_LOWMEM", "") == "1"
os.environ.setdefault("ACESTEP_DTYPE", "bfloat16" if LOWMEM else "float32")
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 4))
os.environ["ACESTEP_INIT_LLM"] = "false"   # we drive the LM ourselves, lazily

# ACE-Step's generation timeout defaults to 600s, sized for GPUs ("GPUs can take
# several minutes" -- generate_music_execute.py:11). On CPU we run at ~15x
# realtime, so anything past ~40s of audio aborts mid-render. Read at import
# time, so it must be set before acestep loads.
os.environ.setdefault("ACESTEP_GENERATION_TIMEOUT", "14400")   # 4 hours

ROOT = os.environ.get("SF_STEPPY_ACESTEP_ROOT", os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get("SF_STEPPY_ACESTEP_CKPT", os.path.join(ROOT, "checkpoints"))
DIT_CONFIG = os.environ.get("SF_STEPPY_ACESTEP_DIT", "acestep-v15-turbo")
LM_MODEL = os.environ.get("SF_STEPPY_ACESTEP_LM", "acestep-5Hz-lm-0.6B")
sys.path.insert(0, ROOT)


def main():
    emit({"event": "loading"})
    try:
        import torch
        torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
        from acestep.handler import AceStepHandler
        from acestep.inference import (GenerationParams, GenerationConfig,
                                       generate_music)
    except Exception as e:
        emit({"event": "error", "msg": f"import failed: {type(e).__name__}: {e}"})
        return

    _dit = {"h": None}

    def get_dit():
        """Load the DiT on FIRST GENERATE, not at startup.

        Deferring it means an auto-lyrics run can write its lyrics in a
        subprocess and let that exit *before* 8GB of DiT is resident, so peak
        is max(2.3, 8) instead of their sum.
        """
        if _dit["h"] is None:
            emit({"event": "stage", "id": 0, "name": "loading model"})
            h = AceStepHandler()
            msg, ok = h.initialize_service(
                project_root=ROOT, config_path=DIT_CONFIG, device="cpu",
                offload_to_cpu=False, use_mlx_dit=False,
                quantization="int8_weight_only" if LOWMEM else None,
            )
            if not ok:
                raise RuntimeError(f"DiT init failed: {msg}")
            _dit["h"] = h
        return _dit["h"]

    sr = 48000

    _LM_SNIPPET = r"""
import json, os, sys
os.environ.setdefault("ACESTEP_DTYPE", "bfloat16")
sys.path.insert(0, sys.argv[1])
import torch; torch.set_num_threads(int(sys.argv[4]))
from acestep.llm_inference import LLMHandler
from acestep.inference import create_sample
h = LLMHandler()
m, ok = h.initialize(checkpoint_dir=sys.argv[2], lm_model_path=sys.argv[3],
                     backend="pt", device="cpu", offload_to_cpu=False, dtype=None)
if not ok:
    print(json.dumps({"error": m})); sys.exit(1)
q = sys.stdin.read()
r = create_sample(h, query=q, instrumental=False, vocal_language=sys.argv[5])
if not getattr(r, "success", False):
    print(json.dumps({"error": getattr(r, "status_message", "create_sample failed")}))
    sys.exit(1)
print(json.dumps({k: getattr(r, k, None)
                  for k in ("lyrics", "bpm", "keyscale", "timesignature")}))
"""

    def write_lyrics(prompt, lang):
        """Run the 5Hz LM in a SHORT-LIVED SUBPROCESS and let it exit.

        Holding the LM (~2.3GB) resident alongside the DiT (~8GB) peaks at
        ~10.3GB, which on a 16GB desktop crosses systemd-oomd's 50%-pressure
        threshold and gets the user's browser killed (observed, not theoretical).
        A subprocess returns every byte to the OS on exit.
        """
        import subprocess
        r = subprocess.run(
            [sys.executable, "-c", _LM_SNIPPET, ROOT, CKPT, LM_MODEL,
             os.environ["OMP_NUM_THREADS"], lang],
            input=prompt, capture_output=True, text=True, timeout=1800)
        out = (r.stdout or "").strip().splitlines()
        for line in reversed(out):
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if "error" in d:
                raise RuntimeError(d["error"])
            return d
        raise RuntimeError(f"LM subprocess produced no result (rc={r.returncode})")

    def spectrogram_png(i16, out_png):
        """Mel spectrogram PNG of the left channel. Mirrors worker.py's output."""
        try:
            import torch
            import torchaudio.transforms as TT
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            wf = torch.from_numpy(i16).float()
            if wf.dim() == 1:
                wf = wf.unsqueeze(0)
            n_fft = 1024
            mel = TT.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=n_fft // 2,
                                    center=True, pad_mode="reflect", power=2.0,
                                    norm="slaney", onesided=True, n_mels=128,
                                    mel_scale="htk")(wf)[0]
            db = 10.0 * np.log10(np.maximum(1e-10, mel.numpy()))
            fig = Figure(figsize=(5, 4), dpi=100)
            FigureCanvasAgg(fig)
            ax = fig.add_subplot()
            ax.imshow(db, origin="lower", aspect="auto", vmin=35, vmax=120)
            ax.set_ylabel("mel bins (log freq)")
            ax.set_xlabel("frame")
            ax.set_title("MelSpectrogram")
            fig.savefig(out_png, bbox_inches="tight")
            return out_png
        except Exception:
            return None

    def handle_generate(cmd):
        gid = cmd.get("id", 0)
        prompt = (cmd.get("prompt") or "").strip()
        duration = int(cmd.get("duration", 30) or 30)
        # Latent size scales with duration, so the 16GB build needs a lower
        # ceiling than the 32GB one. ACE-Step's own constrained decoder does the
        # same thing (set_max_duration, "120 for 2 minutes, 360 for 6 minutes").
        maxdur = int(os.environ.get("SF_STEPPY_MAX_SECONDS", "180" if LOWMEM else "600"))
        duration = max(10, min(maxdur, duration))
        steps = int(cmd.get("steps", 8) or 8)
        seed = int(cmd.get("seed", -1))
        if seed < 0:
            seed = int.from_bytes(os.urandom(4), "little") % (2 ** 31)

        lyrics = cmd.get("lyrics") or ""
        instrumental = bool(cmd.get("instrumental", False))
        bpm = cmd.get("bpm")
        keyscale = cmd.get("keyscale") or ""
        timesig = cmd.get("timesignature") or ""
        lang = cmd.get("language") or "en"
        auto = bool(cmd.get("auto_lyrics", False))
        wrote_lyrics = None

        # --- optional: let the 5Hz LM write the lyrics from the prompt ---
        if auto and not lyrics and not instrumental:
            emit({"event": "stage", "id": gid, "name": "writing lyrics"})
            try:
                s = write_lyrics(prompt, lang)
                if s:
                    # take ONLY the lyrics + musical metadata. The LM also
                    # rewrites the caption and it drifts badly (chillstep ->
                    # "classic video game music"), so the user's prompt wins.
                    lyrics = s.get("lyrics") or ""
                    wrote_lyrics = lyrics
                    bpm = bpm or s.get("bpm")
                    keyscale = keyscale or (s.get("keyscale") or "")
                    timesig = timesig or (s.get("timesignature") or "")
                else:
                    emit({"event": "stage", "id": gid, "name": "lyric writing failed"})
            except Exception as e:
                emit({"event": "error", "msg": f"LM: {type(e).__name__}: {e}"})

        if instrumental:
            lyrics = "[Instrumental]"

        emit({"event": "stage", "id": gid, "name": "generating"})
        t0 = time.time()
        params = GenerationParams(
            task_type="text2music", thinking=False,
            caption=prompt[:512], lyrics=lyrics[:4096],
            bpm=bpm, keyscale=keyscale, timesignature=timesig,
            vocal_language=lang, duration=duration,
            inference_steps=steps, guidance_scale=float(cmd.get("cfg", 1.0) or 1.0),
            seed=seed, instrumental=instrumental,
        )
        tmpdir = os.path.join(os.environ.get("XDG_RUNTIME_DIR", "/tmp"),
                              f"steppy-{gid}")
        os.makedirs(tmpdir, exist_ok=True)
        res = generate_music(get_dit(), None, params=params,
                             config=GenerationConfig(batch_size=1, audio_format="wav"),
                             save_dir=tmpdir)
        if not getattr(res, "success", False):
            emit({"event": "error", "id": gid,
                  "msg": getattr(res, "status_message", "generation failed")})
            return

        src = (res.audios or [{}])[0].get("path")
        if not src or not os.path.isfile(src):
            emit({"event": "error", "id": gid, "msg": "no audio produced"})
            return

        emit({"event": "stage", "id": gid, "name": "writing"})
        with wave.open(src, "rb") as r:
            nch, sw, fr, nfr = r.getnchannels(), r.getsampwidth(), r.getframerate(), r.getnframes()
            raw = r.readframes(nfr)
        out = cmd.get("out") or os.path.join(tmpdir, f"steppy_{gid}.wav")
        with wave.open(out, "wb") as w:
            w.setnchannels(nch); w.setsampwidth(sw); w.setframerate(fr)
            w.writeframes(raw)

        spec = None
        if cmd.get("spectrogram", True):
            a = np.frombuffer(raw, dtype=np.int16)
            if nch > 1:
                a = a.reshape(-1, nch).T[0]
            spec = spectrogram_png(a, out + ".png")

        ev = {"event": "done", "id": gid, "path": out, "spectrogram": spec,
              "seconds": round(nfr / float(fr or 1), 2), "seed": seed,
              "elapsed": round(time.time() - t0, 1)}
        if wrote_lyrics:
            ev["lyrics"] = wrote_lyrics
        emit(ev)
        try:
            os.remove(src)
        except OSError:
            pass

    emit({"event": "ready"})   # ready to accept work; DiT loads on first generate
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except ValueError:
            continue
        c = cmd.get("cmd")
        if c == "ping":
            emit({"event": "ready"})
            continue
        if c != "generate":
            emit({"event": "error", "id": cmd.get("id", 0),
                  "msg": f"unsupported cmd for the Steppy engine: {c!r}"})
            continue
        try:
            handle_generate(cmd)
        except Exception as e:
            import traceback
            emit({"event": "error", "id": cmd.get("id", 0),
                  "msg": f"{type(e).__name__}: {e}",
                  "trace": traceback.format_exc()[-1500:]})


if __name__ == "__main__":
    main()
