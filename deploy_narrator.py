"""
Modal App: adaab-neural-narrator
================================
Provides studio-grade, expressive open-source AI voice narration via Kokoro-82M (Apache 2.0).
Eliminates robotic, flat synthetic speech with human-level prosody, emotional inflection,
charming timbre, and exact word-level timestamps for Wan 2.2 audio-visual sync.
"""

import os
import io
import time
import modal

app = modal.App("adaab-neural-narrator")

def download_kokoro_models():
    """Bakes Kokoro-82M weights and top charming voices directly into the container image."""
    from kokoro import KPipeline
    print("[Image Build] Pre-caching Kokoro pipelines and voices...")
    pa = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    pb = KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M")
    
    # Warm up key charming & expressive voices so they are 100% baked in with zero runtime download
    warmup_voices = ["af_heart", "am_adam", "af_bella", "am_fenrir", "bf_emma", "af_sarah"]
    for v in warmup_voices:
        p = pb if v.startswith("b") else pa
        try:
            for _ in p("Warmup audio synthesis.", voice=v):
                pass
            print(f"[Image Build] Pre-cached voice: {v}")
        except Exception as e:
            print(f"[Image Build] Warning: could not pre-cache voice {v}: {e}")
    print("[Image Build] Pre-caching complete!")


narrator_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg", "espeak-ng", "libsndfile1")
    .pip_install(
        "torch>=2.6.0",
        "torchaudio>=2.6.0",
        "transformers<4.48.0",
        "kokoro>=0.8.4",
        "soundfile",
        "scipy",
        "numpy",
        "huggingface_hub",
    )
    .run_function(download_kokoro_models)
)


@app.cls(
    image=narrator_image,
    timeout=300,
    min_containers=0,
    max_containers=1,
    scaledown_window=45,
)
class NeuralNarratorModel:
    @modal.method()
    def heartbeat(self) -> str:
        return "alive"

    @modal.enter()
    def load_pipeline(self):
        import torch
        from kokoro import KPipeline

        print("[NeuralNarratorModel] Initializing Kokoro-82M neural pipeline...")
        # Initialize US English ('a') and British English ('b') pipelines
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.pipelines = {
            "a": KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M", device=device),
            "b": KPipeline(lang_code="b", repo_id="hexgrad/Kokoro-82M", device=device),
        }
        print(f"[NeuralNarratorModel] Kokoro-82M ready on {device}!")

    @modal.method()
    def synthesize(
        self,
        text: str,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang_code: str = "a",
    ) -> dict:
        """
        Synthesizes natural, expressive speech with exact word-level timing.
        Returns:
          {
            "audio_bytes": bytes (WAV 24kHz mono PCM),
            "duration_sec": float,
            "sample_rate": 24000,
            "word_timestamps": [{"word": str, "start": float, "end": float}],
            "voice": str,
          }
        """
        import io
        import soundfile as sf
        import numpy as np

        text_clean = str(text or "").strip()
        if not text_clean:
            # Return 0.5s of silence
            silence = np.zeros(12000, dtype=np.float32)
            buf = io.BytesIO()
            sf.write(buf, silence, 24000, format="WAV", subtype="PCM_16")
            return {
                "audio_bytes": buf.getvalue(),
                "duration_sec": 0.5,
                "sample_rate": 24000,
                "word_timestamps": [],
                "voice": voice,
            }

        eff_lang = "b" if voice.startswith("b") or lang_code == "b" else "a"
        pipe = self.pipelines.get(eff_lang) or self.pipelines["a"]

        speed = max(0.75, min(1.30, float(speed)))
        generator = pipe(text_clean, voice=voice, speed=speed, split_pattern=r"\n+")

        audio_pieces = []
        word_timestamps = []
        time_offset = 0.0

        for result in generator:
            audio_np = result.audio
            if hasattr(audio_np, "cpu"):
                audio_np = audio_np.cpu().numpy()
            audio_np = np.asarray(audio_np, dtype=np.float32)

            duration = len(audio_np) / 24000.0

            # Extract token / word timestamps if available
            tokens = getattr(result, "tokens", None)
            if tokens:
                for tok in tokens:
                    w = getattr(tok, "text", "")
                    ts_start = getattr(tok, "start_ts", None)
                    ts_end = getattr(tok, "end_ts", None)
                    if w and ts_start is not None and ts_end is not None:
                        w_clean = w.strip()
                        if w_clean:
                            word_timestamps.append({
                                "word": w_clean,
                                "start": round(time_offset + float(ts_start), 3),
                                "end": round(time_offset + float(ts_end), 3),
                            })

            audio_pieces.append(audio_np)
            time_offset += duration

        if not audio_pieces:
            full_audio = np.zeros(24000, dtype=np.float32)
        else:
            full_audio = np.concatenate(audio_pieces)

        # Normalize peak
        max_val = np.max(np.abs(full_audio))
        if max_val > 1e-4:
            full_audio = full_audio * (0.95 / max_val)

        # Encode to WAV 24kHz mono PCM 16-bit
        buf = io.BytesIO()
        sf.write(buf, full_audio, 24000, format="WAV", subtype="PCM_16")
        wav_bytes = buf.getvalue()
        total_dur = len(full_audio) / 24000.0

        return {
            "audio_bytes": wav_bytes,
            "duration_sec": round(total_dur, 3),
            "sample_rate": 24000,
            "word_timestamps": word_timestamps,
            "voice": voice,
        }
