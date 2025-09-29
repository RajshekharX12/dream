from __future__ import annotations
from faster_whisper import WhisperModel
import os
import torch

ASR_MODEL = os.environ.get("ASR_MODEL", "large-v3")
USE_CUDA = os.environ.get("USE_CUDA", "true").lower() == "true"

_device = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"
_compute = "float16" if _device == "cuda" else "int8"

_model: WhisperModel | None = None

def _get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(ASR_MODEL, device=_device, compute_type=_compute)
    return _model

def transcribe(wav_path: str) -> list[dict]:
    model = _get_model()
    segments, _info = model.transcribe(
        wav_path,
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
        language="en",
        condition_on_previous_text=False,
    )
    out = []
    for seg in segments:
        out.append({"start": float(seg.start), "end": float(seg.end), "text": seg.text.strip()})
    return out
