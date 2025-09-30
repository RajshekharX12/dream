# bot.py
# Tele Dubber — English → Hindi dub with optional voice cloning
# aiogram v3 compatible, robust status UI, safe fallbacks

import asyncio
import os
import json
import shutil
import subprocess
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# --- lightweight std / sys metrics ---
import psutil

# --- Telegram / config ---
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import (
    Message,
    CallbackQuery,
    FSInputFile,
    BotCommand,
    Update,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.exceptions import TelegramBadRequest
from dotenv import load_dotenv

# =========================
# CONFIG & BOOTSTRAP
# =========================
load_dotenv(dotenv_path=".env")  # Avoids dotenv AssertionError in REPL/inline

BOT_TOKEN = os.environ.get("BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN missing. Put it in .env")

USE_CUDA = os.environ.get("USE_CUDA", "true").lower() == "true"
MAX_DURATION_MIN = int(os.environ.get("MAX_DURATION_MIN", 20))
ASR_MODEL_ENV = os.environ.get("ASR_MODEL", "large-v3")
ADMIN_IDS = {
    int(x.strip())
    for x in os.environ.get("ADMIN_IDS", "").split(",")
    if x.strip().isdigit()
}
TARGET_MAX_MB = int(os.environ.get("TARGET_MAX_MB", 1900))
COMPRESS_CRF = int(os.environ.get("COMPRESS_CRF", 27))

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# =========================
# LOG & FILE HELPERS
# =========================
def log(msg: str) -> None:
    print(f"[tele-dubber] {msg}", flush=True)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def workdir(user_id: int) -> str:
    p = os.path.join(os.getcwd(), f"_work_{user_id}")
    ensure_dir(p)
    return p

def clean_dir(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

# =========================
# FFmpeg utilities
# =========================
class FFmpegError(RuntimeError):
    pass

def _run_ffmpeg(args: List[str], timeout: Optional[int] = None) -> None:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"] + args
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.decode("utf-8", errors="ignore"))

def _probe_duration(path: str) -> float:
    out = subprocess.check_output(
        [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "format=duration",
            "-of", "json",
            path,
        ]
    )
    data = json.loads(out.decode("utf-8"))
    return max(0.0, float(data["format"]["duration"]))

def extract_audio_wav(video_path: str, out_wav: str, sr: int = 16000) -> str:
    _run_ffmpeg(["-y", "-i", video_path, "-ac", "1", "-ar", str(sr), out_wav])
    return out_wav

def postprocess_audio_inplace(wav_in: str, enable: bool) -> str:
    if not enable:
        return wav_in
    tmp = wav_in + ".tmp.wav"
    # loudnorm + denoise
    _run_ffmpeg([
        "-y", "-i", wav_in,
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11,afftdn=nr=10:nf=-20",
        tmp
    ])
    os.replace(tmp, wav_in)
    return wav_in

def mux_replace_audio(video_path: str, new_audio_wav: str, out_video: str, crf: Optional[int]) -> str:
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
    _run_ffmpeg(args)
    return out_video

def ensure_size_limit(path: str, target_mb: int, crf: int) -> str:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    if size_mb <= target_mb:
        return path
    out2 = path.rsplit(".", 1)[0] + f".crf{crf}.mp4"
    _run_ffmpeg([
        "-y","-i", path,
        "-c:v","libx264","-preset","medium","-crf", str(crf),
        "-c:a","aac","-b:a","128k","-movflags","+faststart",
        out2
    ])
    return out2

# =========================
# Optional heavy deps — lazy import with safe fallback
# =========================
def _torch_cuda_ok() -> bool:
    try:
        import torch  # noqa
        import torch.cuda  # noqa
        return USE_CUDA and torch.cuda.is_available()
    except Exception:
        return False

def _asr_device_and_compute() -> Tuple[str, str]:
    dev = "cuda" if _torch_cuda_ok() else "cpu"
    comp = "float16" if dev == "cuda" else "int8"
    return dev, comp

_ASR_CACHE = {}

def get_asr(model_name: str):
    """
    Returns faster-whisper model or raises a clear RuntimeError if missing.
    """
    try:
        from faster_whisper import WhisperModel
    except Exception as e:
        raise RuntimeError(
            "ASR needs faster-whisper. Install deps first:\n"
            "pip install faster-whisper"
        ) from e

    if model_name not in _ASR_CACHE:
        dev, comp = _asr_device_and_compute()
        _ASR_CACHE[model_name] = WhisperModel(model_name, device=dev, compute_type=comp)
    return _ASR_CACHE[model_name]

def transcribe_segments(wav_path: str, model_name: str, lang: str) -> List[Dict]:
    model = get_asr(model_name)
    language = None if lang == "auto" else lang
    segs, _info = model.transcribe(
        wav_path,
        beam_size=5,
        vad_filter=True,
        word_timestamps=False,
        language=language,
        condition_on_previous_text=False,
    )
    out = []
    for s in segs:
        out.append({"start": float(s.start), "end": float(s.end), "text": s.text.strip()})
    return out

_TRANSLATOR = None

def get_translator():
    global _TRANSLATOR
    if _TRANSLATOR is None:
        try:
            from transformers import pipeline as hf_pipeline
        except Exception as e:
            raise RuntimeError(
                "Translation needs transformers. Install:\n"
                "pip install 'transformers[torch]' sentencepiece"
            ) from e
        _TRANSLATOR = hf_pipeline("translation", model="Helsinki-NLP/opus-mt-en-hi")
    return _TRANSLATOR

def translate_segments_en_hi(segments: List[Dict]) -> List[Dict]:
    if not segments:
        return []
    tr = get_translator()
    texts = [s["text"] for s in segments]
    results = tr(texts, clean_up_tokenization_spaces=True)
    out = []
    for s, r in zip(segments, results):
        x = dict(s)
        x["text_hi"] = r["translation_text"]
        out.append(x)
    return out

class VoiceCloner:
    def __init__(self):
        try:
            import torch  # noqa: F401
            from TTS.api import TTS  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "TTS voice cloning needs coqui-tts. Install:\n"
                "pip install TTS"
            ) from e
        gpu = _torch_cuda_ok()
        self._tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu)
        self.sample_rate = 24000

    def _synth(self, text_hi: str, ref_wav: Optional[str], simple: bool) -> "np.ndarray":
        import numpy as np  # lazy
        if simple:
            wav = self._tts.tts(text=text_hi, speaker=None, language="hi")
        else:
            wav = self._tts.tts(text=text_hi, speaker_wav=ref_wav, language="hi")
        wav = np.asarray(wav, dtype=np.float32)
        m = float(np.max(np.abs(wav))) if len(wav) else 1.0
        return (wav / m) if m > 1.0 else wav

    def synth_to_duration(self, text_hi: str, ref_wav: Optional[str], target_sec: float, simple: bool) -> Tuple["np.ndarray", int]:
        import numpy as np
        import librosa
        wav = self._synth(text_hi, ref_wav, simple)
        cur = max(0.001, len(wav) / self.sample_rate)
        target = max(0.05, float(target_sec))
        rate = cur / target
        stretched = librosa.effects.time_stretch(wav, rate)
        return stretched.astype(np.float32), self.sample_rate

def compose_dubbed_track(
    original_wav: str,
    segments_hi: List[Dict],
    ref_wav: Optional[str],
    simple_voice: bool,
    keep_bg: bool,
) -> Tuple[str, int]:
    import numpy as np
    import soundfile as sf
    import librosa

    y_orig, sr_orig = sf.read(original_wav)
    if y_orig.ndim > 1:
        y_orig = y_orig.mean(axis=1)
    total_sec = len(y_orig) / sr_orig

    cloner = VoiceCloner()
    sr = cloner.sample_rate
    dubbed = np.zeros(int(total_sec * sr) + sr, dtype=np.float32)

    for seg in segments_hi:
        t1, t2 = float(seg["start"]), float(seg["end"])
        dur = max(0.1, t2 - t1)
        text_hi = (seg.get("text_hi") or "").strip()
        if not text_hi:
            continue
        wav, _ = cloner.synth_to_duration(text_hi, ref_wav, dur, simple=simple_voice)
        s = int(t1 * sr)
        e = s + len(wav)
        if e > len(dubbed):
            dubbed = np.concatenate([dubbed, np.zeros(e - len(dubbed), dtype=np.float32)])
        dubbed[s:e] += wav

    if keep_bg:
        if sr_orig != sr:
            bg = librosa.resample(y_orig.astype(np.float32), orig_sr=sr_orig, target_sr=sr)
        else:
            bg = y_orig.astype(np.float32)
        gain = 0.1  # -20dB bed
        L = min(len(dubbed), len(bg))
        dubbed[:L] = dubbed[:L] + gain * bg[:L]

    peak = float(np.max(np.abs(dubbed))) if len(dubbed) else 1.0
    if peak > 1.0:
        dubbed = dubbed / peak

    out_wav = os.path.join(os.path.dirname(original_wav), "dubbed.wav")
    sf.write(out_wav, dubbed, sr)
    return out_wav, sr

def write_srt(segments_hi: List[Dict], out_path: str) -> None:
    def fmt_time(t: float) -> str:
        h = int(t // 3600); t -= h * 3600
        m = int(t // 60); s = t - m * 60
        return f"{h:02}:{m:02}:{s:06.3f}".replace(".", ",")
    lines = []
    for i, seg in enumerate(segments_hi, start=1):
        t1 = fmt_time(float(seg["start"]))
        t2 = fmt_time(float(seg["end"]))
        text = (seg.get("text_hi") or "").strip()
        lines.append(f"{i}\n{t1} --> {t2}\n{text}\n")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

# =========================
# UI state & helpers
# =========================
@dataclass
class Session:
    mode: str = "clone"           # clone | simple
    out: str = "video"            # video | audio | srt
    bgmix: bool = True            # keep ambience low under VO
    compress: bool = False
    asr_model: str = ASR_MODEL_ENV
    lang: str = "en"              # en | auto
    clean_audio: bool = True
    running: Optional[asyncio.Task] = None
    status_id: Optional[int] = None
    max_minutes: int = MAX_DURATION_MIN

SESSIONS: Dict[int, Session] = {}

def _kb(s: Session):
    kb = InlineKeyboardBuilder()
    kb.button(text=("🎭 Clone Voice" if s.mode=="clone" else "🗣️ Simple Hindi"), callback_data="mode:toggle")
    kb.button(text=("🎬 Video" if s.out=="video" else "🎧 Audio" if s.out=="audio" else "🧾 SRT"), callback_data="out:cycle")
    kb.button(text=("🎚️ BG on" if s.bgmix else "🔇 BG off"), callback_data="bg:toggle")
    kb.button(text=("🗜️ Compress on" if s.compress else "🗜️ Compress off"), callback_data="cmp:toggle")
    kb.button(text=f"🧠 ASR: {s.asr_model}", callback_data="asr:cycle")
    kb.button(text=("🌐 Lang: EN" if s.lang=='en' else "🌐 Lang: AUTO"), callback_data="lang:toggle")
    kb.button(text=("🎛️ Clean on" if s.clean_audio else "🎛️ Clean off"), callback_data="clean:toggle")
    kb.button(text="✖️ Cancel job", callback_data="job:cancel")
    kb.adjust(1)
    return kb.as_markup()

async def _status(m: Message, s: Session, text: str):
    try:
        if s.status_id:
            await bot.edit_message_text(m.chat.id, s.status_id, text, reply_markup=_kb(s))
        else:
            sent = await m.answer(text, reply_markup=_kb(s))
            s.status_id = sent.message_id
    except TelegramBadRequest:
        pass
    except Exception:
        pass

async def _final_status(m: Message, s: Session, ok: bool):
    try:
        if s.status_id:
            await bot.edit_message_text(
                chat_id=m.chat.id,
                message_id=s.status_id,
                text="✅ Done!" if ok else "❌ Failed.",
            )
            s.status_id = None
    except Exception:
        pass

def _allowed(uid: int) -> bool:
    return True if not ADMIN_IDS else (uid in ADMIN_IDS)

# =========================
# STARTUP & ERROR HANDLER
# =========================
async def on_startup():
    await bot.set_my_commands([
        BotCommand(command="start", description="Show controls"),
        BotCommand(command="help", description="How to use"),
        BotCommand(command="ping", description="Health & system info"),
        BotCommand(command="stats", description="Current run stats"),
        BotCommand(command="cancel", description="Cancel running job"),
        BotCommand(command="setmax", description="Set max minutes (admin)"),
    ])
    log("Bot commands set.")

async def errors_handler(event: Update, exception: Exception) -> bool:
    log("--- UNHANDLED EXCEPTION ---")
    try:
        log(event.model_dump_json(indent=2))
    except Exception:
        pass
    log(f"{type(exception).__name__}: {exception}")
    # Try to notify user politely
    target = None
    try:
        target = event.message or (event.callback_query.message if event.callback_query else None)
    except Exception:
        target = None
    if target:
        try:
            await bot.send_message(target.chat.id, "❌ Error occurred. Check logs.")
        except Exception:
            pass
    return True  # swallow

# =========================
# COMMANDS
# =========================
@dp.message(CommandStart())
async def cmd_start(m: Message):
    if not _allowed(m.from_user.id):
        await m.answer("🚫 Access denied (admin-only mode).")
        return
    s = SESSIONS.setdefault(m.from_user.id, Session())
    await m.answer(
        "Send me an **English video**; I’ll return a **Hindi dub**.\n\n"
        "Use buttons to switch **Clone/Simple**, **Video/Audio/SRT**, **BG mix**, **Compression**, **ASR**, **Language**, **Clean audio**.",
        reply_markup=_kb(s),
        parse_mode="Markdown"
    )

@dp.message(Command("help"))
async def cmd_help(m: Message):
    await m.answer(
        "• Send an MP4 or a video document (<= your MAX minutes)\n"
        "• Toggle settings from the inline panel\n"
        "• /cancel to stop current job\n"
        "• /ping for health, /stats for session stats\n"
        "• Admin: /setmax 10  (sets max minutes for you)"
    )

@dp.message(Command("ping"))
async def cmd_ping(m: Message):
    gpu = "Yes" if _torch_cuda_ok() else "No"
    mem = psutil.virtual_memory()
    await m.answer(
        f"🏓 Pong\nGPU: {gpu}\nCPUs: {os.cpu_count()}\n"
        f"RAM used: {mem.percent}% of {round(mem.total/1e9,1)} GB"
    )

@dp.message(Command("stats"))
async def cmd_stats(m: Message):
    running = sum(1 for s in SESSIONS.values() if s.running and not s.running.done())
    total = len(SESSIONS)
    mem = psutil.virtual_memory()
    await m.answer(
        "🧮 Stats\n"
        f"- Active users (this run): {total}\n"
        f"- Running jobs: {running}\n"
        f"- RAM used: {mem.percent}% of {round(mem.total/1e9,1)} GB\n"
    )

@dp.message(Command("cancel"))
async def cmd_cancel(m: Message):
    s = SESSIONS.setdefault(m.from_user.id, Session())
    if s.running and not s.running.done():
        s.running.cancel()
        await m.answer("⛔ Canceled your current job.")
    else:
        await m.answer("No running job.")

@dp.message(Command("setmax"))
async def cmd_setmax(m: Message):
    s = SESSIONS.setdefault(m.from_user.id, Session())
    parts = (m.text or "").strip().split()
    if len(parts) >= 2 and parts[1].isdigit():
        val = int(parts[1])
        s.max_minutes = max(1, min(180, val))
        await m.answer(f"✅ Max duration set to {s.max_minutes} minutes for you.")
    else:
        await m.answer("Usage: /setmax 15  (minutes)")

# =========================
# INLINE TOGGLES
# =========================
@dp.callback_query(F.data.startswith("mode:"))
async def cb_mode(c: CallbackQuery):
    s = SESSIONS.setdefault(c.from_user.id, Session())
    s.mode = "simple" if s.mode == "clone" else "clone"
    await c.message.edit_reply_markup(reply_markup=_kb(s))
    await c.answer(f"Mode: {s.mode}")

@dp.callback_query(F.data.startswith("out:"))
async def cb_out(c: CallbackQuery):
    s = SESSIONS.setdefault(c.from_user.id, Session())
    s.out = "audio" if s.out == "video" else ("srt" if s.out == "audio" else "video")
    await c.message.edit_reply_markup(reply_markup=_kb(s))
    await c.answer(f"Output: {s.out}")

@dp.callback_query(F.data.startswith("bg:"))
async def cb_bg(c: CallbackQuery):
    s = SESSIONS.setdefault(c.from_user.id, Session())
    s.bgmix = not s.bgmix
    await c.message.edit_reply_markup(reply_markup=_kb(s))
    await c.answer(f"BG mix: {'on' if s.bgmix else 'off'}")

@dp.callback_query(F.data.startswith("cmp:"))
async def cb_cmp(c: CallbackQuery):
    s = SESSIONS.setdefault(c.from_user.id, Session())
    s.compress = not s.compress
    await c.message.edit_reply_markup(reply_markup=_kb(s))
    await c.answer(f"Compression: {'on' if s.compress else 'off'}")

@dp.callback_query(F.data.startswith("asr:"))
async def cb_asr(c: CallbackQuery):
    s = SESSIONS.setdefault(c.from_user.id, Session())
    order = ["base", "small", "large-v3"]
    try:
        i = order.index(s.asr_model)
    except ValueError:
        i = 2
    s.asr_model = order[(i + 1) % len(order)]
    await c.message.edit_reply_markup(reply_markup=_kb(s))
    await c.answer(f"ASR: {s.asr_model}")

@dp.callback_query(F.data.startswith("lang:"))
async def cb_lang(c: CallbackQuery):
    s = SESSIONS.setdefault(c.from_user.id, Session())
    s.lang = "auto" if s.lang == "en" else "en"
    await c.message.edit_reply_markup(reply_markup=_kb(s))
    await c.answer(f"Lang: {s.lang}")

@dp.callback_query(F.data.startswith("clean:"))
async def cb_clean(c: CallbackQuery):
    s = SESSIONS.setdefault(c.from_user.id, Session())
    s.clean_audio = not s.clean_audio
    await c.message.edit_reply_markup(reply_markup=_kb(s))
    await c.answer(f"Clean: {'on' if s.clean_audio else 'off'}")

@dp.callback_query(F.data.startswith("job:cancel"))
async def cb_cancel(c: CallbackQuery):
    s = SESSIONS.setdefault(c.from_user.id, Session())
    if s.running and not s.running.done():
        s.running.cancel()
        await c.answer("Canceled.")
    else:
        await c.answer("No job running.")

# =========================
# CORE VIDEO/AUDIO HANDLER
# =========================
@dp.message(F.video | F.document)
async def on_media(m: Message):
    if not _allowed(m.from_user.id):
        await m.reply("🚫 Access denied (admin-only).")
        return

    s = SESSIONS.setdefault(m.from_user.id, Session())

    # duration guard for videos
    vid_dur = m.video.duration if m.video and m.video.duration else 0
    if vid_dur and vid_dur > s.max_minutes * 60:
        await m.reply(f"Too long. Max {s.max_minutes} minutes. Use /setmax to change for yourself.")
        return

    if s.running and not s.running.done():
        await m.reply("You already have a running job. Use /cancel to stop it.")
        return

    wd = workdir(m.from_user.id)
    await asyncio.to_thread(clean_dir, wd)
    ensure_dir(wd)

    # choose filename
    ext = "mp4"
    if m.document and m.document.file_name and "." in m.document.file_name:
        ext = m.document.file_name.rsplit(".", 1)[-1].lower()
        if ext not in ("mp4", "mkv", "mov", "mp3", "wav", "m4a", "aac"):
            ext = "mp4"  # default

    in_path = os.path.join(wd, f"input.{ext}")

    # Download
    await _status(m, s, "⬇️ Downloading…")
    try:
        if m.video:
            await bot.download(m.v
