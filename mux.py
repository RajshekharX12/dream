from __future__ import annotations
import numpy as np
import soundfile as sf
from typing import List, Dict
from voice_clone import VoiceCloner, save_wav
from ffmpeg_utils import mux_replace_audio


def compose_dubbed_track(
    original_wav: str,
    segments: List[Dict],
    ref_speaker_wav: str,
    out_wav: str,
    cloner: VoiceCloner,
) -> str:
    """Generate a Hindi-dubbed mono wav aligned to original timings.
    segments: list of {start, end, text_hi}
    """
    y_orig, sr_orig = sf.read(original_wav)
    if y_orig.ndim > 1:
        y_orig = y_orig.mean(axis=1)
    total_seconds = len(y_orig) / sr_orig

    sr = cloner.sample_rate
    dubbed = np.zeros(int(total_seconds * sr) + sr, dtype=np.float32)

    for seg in segments:
        start_s = float(seg["start"]) ; end_s = float(seg["end"]) ; dur = max(0.1, end_s - start_s)
        text_hi = (seg.get("text_hi") or "").strip()
        if not text_hi:
            continue
        wav, _ = cloner.synthesize_to_duration(text_hi, ref_speaker_wav, target_seconds=dur)
        start_idx = int(start_s * sr)
        end_idx = start_idx + len(wav)
        if end_idx > len(dubbed):
            pad = end_idx - len(dubbed)
            dubbed = np.concatenate([dubbed, np.zeros(pad, dtype=np.float32)])
        dubbed[start_idx:end_idx] += wav

    peak = np.max(np.abs(dubbed))
    if peak > 1.0:
        dubbed = dubbed / peak

    save_wav(out_wav, dubbed, sr)
    return out_wav


def replace_video_audio(video_path: str, dubbed_wav: str, out_video: str) -> str:
    return mux_replace_audio(video_path, dubbed_wav, out_video)
