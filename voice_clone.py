from __future__ import annotations
import numpy as np
import soundfile as sf
import librosa
import os
import torch
from TTS.api import TTS

USE_CUDA = os.environ.get("USE_CUDA", "true").lower() == "true"
_device_gpu = bool(USE_CUDA and torch.cuda.is_available())

class VoiceCloner:
    """XTTS-v2 Hindi voice cloning with ~6s reference clip."""
    def __init__(self, model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2") -> None:
        self.tts = TTS(model_name, gpu=_device_gpu)
        self.sample_rate = 24000  # XTTS default

    def synthesize(self, text_hi: str, ref_wav: str) -> np.ndarray:
        wav = self.tts.tts(text=text_hi, speaker_wav=ref_wav, language="hi")
        wav = np.asarray(wav, dtype=np.float32)
        if len(wav) > 0:
            peak = np.max(np.abs(wav))
            if peak > 1.0:
                wav = wav / peak
        return wav

    def synthesize_to_duration(self, text_hi: str, ref_wav: str, target_seconds: float) -> tuple[np.ndarray, int]:
        wav = self.synthesize(text_hi, ref_wav)
        cur = max(0.001, len(wav) / self.sample_rate)
        target = max(0.05, target_seconds)
        rate = cur / target
        stretched = librosa.effects.time_stretch(wav, rate)
        return stretched.astype(np.float32), self.sample_rate

def save_wav(path: str, wav: np.ndarray, sr: int) -> None:
    sf.write(path, wav, sr)
