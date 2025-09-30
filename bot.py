import asyncio
import os
import math
import json
import subprocess
import shutil
import time
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
import psutil

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.utils.keyboard import InlineKeyboardBuilder

# ---------- CONFIG ----------
load_dotenv()
BOT_TOKEN = os.environ.get("BOT_TOKEN")
USE_CUDA = os.environ.get("USE_CUDA", "true").lower() == "true"
MAX_DURATION_MIN = int(os.environ.get("MAX_DURATION_MIN", 20))
ASR_MODEL_ENV = os.environ.get("ASR_MODEL", "large-v3")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN missing. Put it in .env")

# ---------- GLOBALS ----------
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# Per-user runtime config + job tracking
@dataclass
class SessionState:
    mode: str = "clone"         # clone | simple
    out: str = "video"          # video | audio | srt
    bgmix: bool = True          # keep original ambience mixed low
    compress: bool = False      # re-encode with CRF 26
    asr_model: str = ASR_MODEL_ENV  # base | small | large-v3
    running_task: Optional[asyncio.Task] = None
    status_msg_id: Optional[int] = None

SESSIONS: Dict[int, SessionState] = {}

# ---------- UTIL: LOGGING ----------
def log(msg: str):
    print(f"[tele-hindi-dubber] {msg}", flush=True)

# ---------- UTIL: FILES ----------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def user_workdir(user_id: int) -> str:
    path = os.path.join(os.getcwd(), f"_work_{user_id}")
    ensure_dir(path)
    return path

def clean_dir(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

# ---------- UTIL: FFMPEG ----------
class FFmpegError(RuntimeError):
    pass

def run_ffmpeg(args: List[str], timeout: Optional[int] = None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"] + args
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.decode("utf-8", errors="ignore"))

def run_ffprobe_get_duration(path: str) -> float:
    cmd = [
        "ffprobe","-v","error","-select_streams","v:0","-show_entries",
        "format=duration","-of","json", path
    ]
    out = subprocess.check_output(cmd)
    data = json.loads(out.decode("utf-8"))
    dur = float(data["format"]["duration"])
    return max(0.0, dur)

def extract_audio_wav(video_path: str, out_wav: str, sr: int = 16000, mono: bool = True) -> str:
    args = ["-y", "-i", video_path]
    if mono:
        args += ["-ac", "1"]
    args += ["-ar", str(sr), out_wav]
    run_ffmpeg(args)
    return out_wav

def ffmpeg_extract(in_wav: str, out_wav: str, start: float, end: float):
    duration = max(0.1, end - start)
    run_ffmpeg(["-y", "-ss", f"{start}", "-t", f"{duration}", "-i", in_wav, out_wav])

def mux_replace_audio(video_path: str, new_audio_wav: str, out_video: str, crf: Optional[int] = None) -> str:
    # If crf is None, copy video stream; else re-encode video for size control
    if crf is None:
        args = [
            "-y","-i", video_path,"-i", new_audio_wav,
            "-map","0:v:0","-map","1:a:0",
            "-c:v","copy","-c:a","aac","-b:a","192k","-shortest", out_video
        ]
    else:
        args = [
            "-y","-i", video_path,"-i", new_audio_wav,
            "-map","0:v:0","-map","1:a:0",
            "-c:v","libx264","-preset","medium","-crf", str(crf),
            "-c:a","aac","-b:a","160k","-shortest", out_video
        ]
    run_ffmpeg(args)
    return out_video

# ---------- ASR: faster-whisper ----------
# Install at runtime:
# pip install faster-whisper
from faster_whisper import WhisperModel
import torch

def asr_device_and_compute() -> Tuple[str, str]:
    dev = "cuda" if USE_CUDA and torch.cuda.is_available() else "cpu"
    comp = "float16" if dev == "cuda" else "int8"
    return dev, comp

_ASR_CACHE: Dict[str, WhisperModel] = {}

def get_asr(model_name: str) -> WhisperModel:
    if model_name not in _ASR_CACHE:
        dev, comp = asr_device_and_compute()
        _ASR_CACHE[model_name] = WhisperModel(model_name, device=dev, compute_type=comp)
    return _ASR_CACHE[model_name]

def transcribe_segments(wav_path: str, model_name: str) -> List[Dict]:
    model = get_asr(model_name)
    segs, _info = model.transcribe(
        wav_path,
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
        language="en",
        condition_on_previous_text=False,
    )
    out = []
    for s in segs:
        out.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
    return out

# ---------- Translation: HF pipeline (en->hi) ----------
from transformers import pipeline as hf_pipeline
_TRANSLATOR = None

def get_translator():
    global _TRANSLATOR
    if _TRANSLATOR is None:
        _TRANSLATOR = hf_pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
    return _TRANSLATOR

def translate_segments_en_hi(segments: List[Dict]) -> List[Dict]:
    if not segments:
        return []
    translator = get_translator()
    texts = [s["text"] for s in segments]
    results = translator(texts, clean_up_tokenization_spaces=True)
    hi_texts = [r["translation_text"] for r in results]
    out = []
    for s, t in zip(segments, hi_texts):
        x = dict(s)
        x["text_hi"] = t
        out.append(x)
    return out

# ---------- TTS Voice Clone: Coqui XTTS-v2 ----------
# pip install TTS
from TTS.api import TTS

class VoiceCloner:
    def __init__(self):
        gpu = bool(USE_CUDA and torch.cuda.is_available())
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu)
        # XTTS default sr ~24000
        self.sample_rate = 24000

    def synth(self, text_hi: str, ref_wav: Optional[str], simple: bool=False) -> np.ndarray:
        if simple:
            wav = self.tts.tts(text=text_hi, speaker=None, language="hi")
        else:
            wav = self.tts.tts(text=text_hi, speaker_wav=ref_wav, language="hi")
        wav = np.asarray(wav, dtype=np.float32)
        # normalize
        m = np.max(np.abs(wav)) if len(wav) else 1.0
        return (wav / m) if m > 1.0 else wav

    def synth_to_duration(self, text_hi: str, ref_wav: Optional[str], target_sec: float, simple: bool=False) -> Tuple[np.ndarray, int]:
        wav = self.synth(text_hi, ref_wav, simple=simple)
        cur = max(0.001, len(wav) / self.sample_rate)
        target = max(0.05, target_sec)
        rate = cur / target
        stretched = librosa.effects.time_stretch(wav, rate)
        return stretched.astype(np.float32), self.sample_rate

# ---------- Compose Dubbed Track ----------
def compose_dubbed_track(
    original_wav: str,
    segments_hi: List[Dict],
    ref_wav: str,
    cloner: VoiceCloner,
    simple_voice: bool,
    keep_bg: bool
) -> Tuple[str, int]:
    y_orig, sr_orig = sf.read(original_wav)
    if y_orig.ndim > 1:
        y_orig = y_orig.mean(axis=1)
    total_sec = len(y_orig) / sr_orig

    sr = cloner.sample_rate
    dubbed = np.zeros(int(total_sec * sr) + sr, dtype=np.float32)

    for seg in segments_hi:
        start_s = float(seg["start"])
        end_s = float(seg["end"])
        dur = max(0.1, end_s - start_s)
        text_hi = (seg.get("text_hi") or "").strip()
        if not text_hi:
            continue
        wav, _ = cloner.synth_to_duration(text_hi, ref_wav, dur, simple=simple_voice)
        start_idx = int(start_s * sr)
        end_idx = start_idx + len(wav)
        if end_idx > len(dubbed):
            pad = end_idx - len(dubbed)
            dubbed = np.concatenate([dubbed, np.zeros(pad, dtype=np.float32)])
        dubbed[start_idx:end_idx] += wav

    # mix low-level ambience if requested
    if keep_bg:
        # resample original to sr if needed
        if sr_orig != sr:
            y_res = librosa.resample(y_orig.astype(np.float32), orig_sr=sr_orig, target_sr=sr)
        else:
            y_res = y_orig.astype(np.float32)
        # -20 dB mix (~0.1 gain)
        gain = 0.1
        L = min(len(dubbed), len(y_res))
        dubbed[:L] = dubbed[:L] + gain * y_res[:L]

    # normalize
    peak = np.max(np.abs(dubbed)) if len(dubbed) else 1.0
    if peak > 1.0:
        dubbed = dubbed / peak

    out_wav = os.path.join(os.path.dirname(original_wav), "dubbed.wav")
    sf.write(out_wav, dubbed, sr)
    return out_wav, sr

# ---------- SRT writer ----------
def write_srt(segments_hi: List[Dict], out_path: str):
    def fmt_time(t: float) -> str:
        h = int(t // 3600); t -= h*3600
        m = int(t // 60); s = t - m*60
        return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")
    lines = []
    for i, seg in enumerate(segments_hi, start=1):
        t1 = fmt_time(float(seg["start"]))
        t2 = fmt_time(float(seg["end"]))
        text = (seg.get("text_hi") or "").strip()
        lines.append(f"{i}\n{t1} --> {t2}\n{text}\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# ---------- UI (inline keyboard) ----------
def build_kb(st: SessionState):
    kb = InlineKeyboardBuilder()
    # mode
    kb.button(text=("🎭 Clone Voice" if st.mode=="clone" else "🗣️ Simple Hindi"), callback_data="mode:toggle")
    # output
    kb.button(text=("🎬 Video" if st.out=="video" else "🎧 Audio" if st.out=="audio" else "🧾 SRT"), callback_data="out:cycle")
    # bg mix
    kb.button(text=("🎚️ BG on" if st.bgmix else "🔇 BG off"), callback_data="bg:toggle")
    # compress
    kb.button(text=("🗜️ Compress on" if st.compress else "🗜️ Compress off"), callback_data="cmp:toggle")
    # asr model
    kb.button(text=f"🧠 ASR: {st.asr_model}", callback_data="asr:cycle")
    # cancel
    kb.button(text="✖️ Cancel job", callback_data="job:cancel")
    kb.adjust(1)
    return kb.as_markup()

async def update_status(m: Message, st: SessionState, text: str):
    try:
        if st.status_msg_id:
            await bot.edit_message_text(chat_id=m.chat.id, message_id=st.status_msg_id, text=text, reply_markup=build_kb(st))
        else:
            sent = await m.answer(text, reply_markup=build_kb(st))
            st.status_msg_id = sent.message_id
    except Exception:
        pass

# ---------- HANDLERS ----------
@dp.message(CommandStart())
async def start_cmd(m: Message):
    SESSIONS[m.from_user.id] = SessionState()
    await m.answer(
        "Send me an English video and I'll return a **Hindi dub**.\n\n"
        "Use the buttons below to pick **Clone/Simple voice**, **Video/Audio/SRT**, "
        "**Background mix**, **Compression**, and **ASR model**.",
        reply_markup=build_kb(SESSIONS[m.from_user.id]),
        parse_mode="Markdown"
    )

@dp.message(Command("stats"))
async def stats_cmd(m: Message):
    running = sum(1 for s in SESSIONS.values() if s.running_task and not s.running_task.done())
    total = len(SESSIONS)
    mem = psutil.virtual_memory()
    txt = (
        f"🧮 Stats\n"
        f"- Active users (this run): {total}\n"
        f"- Running jobs: {running}\n"
        f"- RAM used: {mem.percent}% of {round(mem.total/1e9,1)} GB\n"
    )
    await m.answer(txt)

@dp.message(Command("cancel"))
async def cancel_cmd(m: Message):
    st = SESSIONS.setdefault(m.from_user.id, SessionState())
    if st.running_task and not st.running_task.done():
        st.running_task.cancel()
        await m.answer("⛔ Canceled your current job.")
    else:
        await m.answer("No running job to cancel.")

@dp.callback_query(F.data.startswith("mode:"))
async def cb_mode(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    st.mode = "simple" if st.mode == "clone" else "clone"
    await c.message.edit_reply_markup(build_kb(st))
    await c.answer(f"Mode: {st.mode}")

@dp.callback_query(F.data.startswith("out:"))
async def cb_out(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    st.out = "audio" if st.out=="video" else ("srt" if st.out=="audio" else "video")
    await c.message.edit_reply_markup(build_kb(st))
    await c.answer(f"Output: {st.out}")

@dp.callback_query(F.data.startswith("bg:"))
async def cb_bg(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    st.bgmix = not st.bgmix
    await c.message.edit_reply_markup(build_kb(st))
    await c.answer(f"BG mix: {'on' if st.bgmix else 'off'}")

@dp.callback_query(F.data.startswith("cmp:"))
async def cb_cmp(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    st.compress = not st.compress
    await c.message.edit_reply_markup(build_kb(st))
    await c.answer(f"Compression: {'on' if st.compress else 'off'}")

@dp.callback_query(F.data.startswith("asr:"))
async def cb_asr(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    order = ["base","small","large-v3"]
    try:
        i = order.index(st.asr_model)
    except ValueError:
        i = 2
    st.asr_model = order[(i+1)%len(order)]
    await c.message.edit_reply_markup(build_kb(st))
    await c.answer(f"ASR: {st.asr_model}")

@dp.callback_query(F.data.startswith("job:cancel"))
async def cb_cancel(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    if st.running_task and not st.running_task.done():
        st.running_task.cancel()
        await c.answer("Canceled.")
    else:
        await c.answer("No job running.")

# Accept video or video document
@dp.message(F.video | F.document)
async def on_video(m: Message):
    st = SESSIONS.setdefault(m.from_user.id, SessionState())

    # Reject too long
    vid_dur = 0
    if m.video:
        vid_dur = m.video.duration or 0
    if vid_dur and vid_dur > MAX_DURATION_MIN * 60:
        await m.reply(f"Video too long. Max {MAX_DURATION_MIN} minutes (set MAX_DURATION_MIN in .env).")
        return

    # Only one job per user
    if st.running_task and not st.running_task.done():
        await m.reply("You already have a running job. Use /cancel to stop it.")
        return

    # Download
    work = user_workdir(m.from_user.id)
    ensure_dir(work)
    in_path = os.path.join(work, "input.mp4")

    await update_status(m, st, "⬇️ Downloading video…")
    file_obj = m.video or m.document
    # Best compatibility across aiogram versions:
    try:
        await bot.download(file=file_obj, destination=in_path)  # aiogram >=3.0
    except Exception:
        f = await bot.get_file(file_obj.file_id)
        # Fallback: use getFile URL via Bot method
        await bot.download_file(f.file_path, in_path)

    # Spawn processing task
    st.running_task = asyncio.create_task(process_job(m, st, in_path))
    try:
        await st.running_task
    except asyncio.CancelledError:
        await update_status(m, st, "❌ Job canceled.")
    finally:
        st.running_task = None

# ---------- CORE PIPELINE ----------
async def process_job(m: Message, st: SessionState, in_path: str):
    work = os.path.dirname(in_path)
    try:
        await update_status(m, st, "🎵 Extracting audio…")
        wav16k = os.path.join(work, "original.wav")
        await asyncio.to_thread(extract_audio_wav, in_path, wav16k, 16000, True)

        await update_status(m, st, f"🧠 Transcribing (ASR={st.asr_model})…")
        segments = await asyncio.to_thread(transcribe_segments, wav16k, st.asr_model)
        if not segments:
            await update_status(m, st, "I couldn't detect any English speech.")
            return

        await update_status(m, st, "🌐 Translating to Hindi…")
        segments_hi = await asyncio.to_thread(translate_segments_en_hi, segments)

        # Build reference voice sample (6–8s from first speech)
        s0 = segments[0]
        ref_start = max(0.0, float(s0["start"]))
        ref_dur = max(6.0, min(8.0, float(s0["end"]) - ref_start))
        ref_path = os.path.join(work, "ref.wav")
        await asyncio.to_thread(ffmpeg_extract, wav16k, ref_path, ref_start, ref_start + ref_dur)

        await update_status(m, st, "🗣️ Synthesizing Hindi voice…")
        cloner = VoiceCloner()
        simple = (st.mode != "clone")
        out_wav, _sr = await asyncio.to_thread(
            compose_dubbed_track,
            wav16k,
            segments_hi,
            ref_path,
            cloner,
            simple,
            st.bgmix
        )

        if st.out == "audio":
            await update_status(m, st, "📤 Sending audio…")
            await m.answer_document(FSInputFile(out_wav), caption="Hindi dub (audio)")
            await finalize_status(m, st)
            return

        if st.out == "srt":
            await update_status(m, st, "🧾 Writing SRT…")
            srt_path = os.path.join(work, "subtitles_hi.srt")
            await asyncio.to_thread(write_srt, segments_hi, srt_path)
            await update_status(m, st, "📤 Sending SRT…")
            await m.answer_document(FSInputFile(srt_path), caption="Hindi subtitles (.srt)")
            await finalize_status(m, st)
            return

        # Replace audio in video
        await update_status(m, st, "🎬 Muxing video…")
        out_video = os.path.join(work, "dubbed.mp4")
        crf = 26 if st.compress else None
        await asyncio.to_thread(mux_replace_audio, in_path, out_wav, out_video, crf)

        size = os.path.getsize(out_video)
        if size > 2 * 1024 * 1024 * 1024:
            await update_status(m, st, "⚠️ Output exceeds Telegram 2GB limit. Sending audio-only instead.")
            await m.answer_document(FSInputFile(out_wav), caption="Hindi dub (audio)")
        else:
            await update_status(m, st, "📤 Uploading video…")
            await m.answer_video(FSInputFile(out_video), caption="Hindi dub (cloned voice)")

        await finalize_status(m, st)

    finally:
        # keep workdir for now; comment next line if you want auto-clean
        # clean_dir(work)
        pass

async def finalize_status(m: Message, st: SessionState):
    await update_status(m, st, "✅ Done. Send another video anytime.")

# ---------- MAIN ----------
def main():
    log("Starting bot…")
    dp.run_polling(bot)

if __name__ == "__main__":
    main()
