cat > bot.py <<'PY'
import asyncio
import os
import json
import subprocess
import shutil
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import soundfile as sf
import librosa
import psutil

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, CallbackQuery, FSInputFile, BotCommand
from aiogram.utils.keyboard import InlineKeyboardBuilder

# ==================== CONFIG ====================
load_dotenv()
BOT_TOKEN = os.environ.get("BOT_TOKEN")
USE_CUDA = os.environ.get("USE_CUDA", "true").lower() == "true"
MAX_DURATION_MIN = int(os.environ.get("MAX_DURATION_MIN", 20))
ASR_MODEL_ENV = os.environ.get("ASR_MODEL", "large-v3")
ADMIN_IDS = set(
    int(x.strip()) for x in os.environ.get("ADMIN_IDS", "").split(",") if x.strip().isdigit()
)
TARGET_MAX_MB = int(os.environ.get("TARGET_MAX_MB", 1900))
COMPRESS_CRF = int(os.environ.get("COMPRESS_CRF", 27))

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN missing. Put it in .env")

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

# ==================== STATE ====================
@dataclass
class SessionState:
    mode: str = "clone"            # clone | simple
    out: str = "video"             # video | audio | srt
    bgmix: bool = True             # keep original ambience mixed low
    compress: bool = False         # user-chosen extra compression
    asr_model: str = ASR_MODEL_ENV # base | small | large-v3
    lang: str = "en"               # en | auto (ASR language)
    clean_audio: bool = True       # loudnorm + denoise
    running_task: Optional[asyncio.Task] = None
    status_msg_id: Optional[int] = None

SESSIONS: Dict[int, SessionState] = {}

# ==================== LOG/FILES ====================
def log(msg: str):
    print(f"[tele-hindi-dubber] {msg}", flush=True)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def user_workdir(user_id: int) -> str:
    path = os.path.join(os.getcwd(), f"_work_{user_id}")
    ensure_dir(path)
    return path

def clean_dir(path: str):
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)

# ==================== FFMPEG UTILS ====================
class FFmpegError(RuntimeError):
    pass

def run_ffmpeg(args: List[str], timeout: Optional[int] = None):
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"] + args
    proc = subprocess.run(cmd, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        raise FFmpegError(proc.stderr.decode("utf-8", errors="ignore"))

def run_ffprobe_get_duration(path: str) -> float:
    out = subprocess.check_output(
        ["ffprobe","-v","error","-select_streams","v:0","-show_entries","format=duration","-of","json", path]
    )
    data = json.loads(out.decode("utf-8"))
    return max(0.0, float(data["format"]["duration"]))

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

def postprocess_audio_inplace(wav_in: str, clean_audio: bool) -> str:
    if not clean_audio:
        return wav_in
    tmp = wav_in + ".tmp.wav"
    run_ffmpeg([
        "-y","-i", wav_in,
        "-af","loudnorm=I=-16:TP=-1.5:LRA=11,afftdn=nr=10:nf=-20",
        tmp
    ])
    os.replace(tmp, wav_in)
    return wav_in

def ensure_size_limit(path: str, target_mb: int, crf: int) -> str:
    size_mb = os.path.getsize(path) / (1024*1024)
    if size_mb <= target_mb:
        return path
    out2 = path.replace(".mp4", f".crf{crf}.mp4")
    run_ffmpeg([
        "-y","-i", path,
        "-c:v","libx264","-preset","medium","-crf", str(crf),
        "-c:a","aac","-b:a","128k","-movflags","+faststart",
        out2
    ])
    return out2

# ==================== ASR (faster-whisper) ====================
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

# ==================== Translation (en->hi) ====================
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

# ==================== TTS / Voice Cloning (XTTS) ====================
from TTS.api import TTS

class VoiceCloner:
    def __init__(self):
        gpu = bool(USE_CUDA and torch.cuda.is_available())
        self.tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=gpu)
        self.sample_rate = 24000

    def synth(self, text_hi: str, ref_wav: Optional[str], simple: bool=False) -> np.ndarray:
        if simple:
            wav = self.tts.tts(text=text_hi, speaker=None, language="hi")
        else:
            wav = self.tts.tts(text=text_hi, speaker_wav=ref_wav, language="hi")
        wav = np.asarray(wav, dtype=np.float32)
        m = np.max(np.abs(wav)) if len(wav) else 1.0
        return (wav / m) if m > 1.0 else wav

    def synth_to_duration(self, text_hi: str, ref_wav: Optional[str], target_sec: float, simple: bool=False) -> Tuple[np.ndarray, int]:
        wav = self.synth(text_hi, ref_wav, simple=simple)
        cur = max(0.001, len(wav) / self.sample_rate)
        target = max(0.05, target_sec)
        rate = cur / target
        stretched = librosa.effects.time_stretch(wav, rate)
        return stretched.astype(np.float32), self.sample_rate

# ==================== Compose Dubbed Track ====================
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

    if keep_bg:
        if sr_orig != sr:
            y_res = librosa.resample(y_orig.astype(np.float32), orig_sr=sr_orig, target_sr=sr)
        else:
            y_res = y_orig.astype(np.float32)
        gain = 0.1  # -20 dB
        L = min(len(dubbed), len(y_res))
        dubbed[:L] = dubbed[:L] + gain * y_res[:L]

    peak = np.max(np.abs(dubbed)) if len(dubbed) else 1.0
    if peak > 1.0:
        dubbed = dubbed / peak

    out_wav = os.path.join(os.path.dirname(original_wav), "dubbed.wav")
    sf.write(out_wav, dubbed, sr)
    return out_wav, sr

# ==================== SRT ====================
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

# ==================== UI (Inline Keyboard) ====================
def build_kb(st: SessionState):
    kb = InlineKeyboardBuilder()
    kb.button(text=("🎭 Clone Voice" if st.mode=="clone" else "🗣️ Simple Hindi"), callback_data="mode:toggle")
    kb.button(text=("🎬 Video" if st.out=="video" else "🎧 Audio" if st.out=="audio" else "🧾 SRT"), callback_data="out:cycle")
    kb.button(text=("🎚️ BG on" if st.bgmix else "🔇 BG off"), callback_data="bg:toggle")
    kb.button(text=("🗜️ Compress on" if st.compress else "🗜️ Compress off"), callback_data="cmp:toggle")
    kb.button(text=f"🧠 ASR: {st.asr_model}", callback_data="asr:cycle")
    kb.button(text=("🌐 Lang: EN" if st.lang=='en' else "🌐 Lang: AUTO"), callback_data="lang:toggle")
    kb.button(text=("🎛️ Clean on" if st.clean_audio else "🎛️ Clean off"), callback_data="clean:toggle")
    kb.button(text="✖️ Cancel job", callback_data="job:cancel")
    kb.adjust(1)
    return kb.as_markup()

async def update_status(m: Message, st: SessionState, text: str):
    try:
        if st.status_msg_id:
            await bot.edit_message_text(
                chat_id=m.chat.id,
                message_id=st.status_msg_id,
                text=text,
                reply_markup=build_kb(st)
            )
        else:
            sent = await m.answer(text, reply_markup=build_kb(st))
            st.status_msg_id = sent.message_id
    except Exception:
        pass

# ==================== ACCESS ====================
def allowed_user(user_id: int) -> bool:
    if not ADMIN_IDS:
        return True
    return user_id in ADMIN_IDS

# ==================== STARTUP ====================
@dp.startup.register
async def on_startup():
    await bot.set_my_commands([
        BotCommand(command="start", description="Start / show controls"),
        BotCommand(command="stats", description="Show stats"),
        BotCommand(command="cancel", description="Cancel current job"),
        BotCommand(command="ping", description="Ping / system info"),
    ])

# ==================== HANDLERS ====================
@dp.message(CommandStart())
async def start_cmd(m: Message):
    if not allowed_user(m.from_user.id):
        await m.answer("🚫 Access denied. This bot is in admin-only mode.")
        return
    SESSIONS[m.from_user.id] = SessionState()
    await m.answer(
        "Send me an English video and I'll return a **Hindi dub**.\n\n"
        "Use the buttons to pick **Clone/Simple voice**, **Video/Audio/SRT**, "
        "**Background mix**, **Compression**, **ASR model**, **Language**, **Audio clean-up**.",
        reply_markup=build_kb(SESSIONS[m.from_user.id]),
        parse_mode="Markdown"
    )

@dp.message(Command("ping"))
async def ping_cmd(m: Message):
    import torch
    gpu = torch.cuda.is_available()
    cpu_count = os.cpu_count()
    mem = psutil.virtual_memory()
    await m.answer(
        f"🏓 Pong\nGPU: {'Yes' if gpu else 'No'}\nCPUs: {cpu_count}\nRAM used: {mem.percent}% of {round(mem.total/1e9,1)} GB"
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

# ---- callbacks (IMPORTANT: use reply_markup=...) ----
@dp.callback_query(F.data.startswith("mode:"))
async def cb_mode(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    st.mode = "simple" if st.mode == "clone" else "clone"
    await c.message.edit_reply_markup(reply_markup=build_kb(st))
    await c.answer(f"Mode: {st.mode}")

@dp.callback_query(F.data.startswith("out:"))
async def cb_out(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    st.out = "audio" if st.out=="video" else ("srt" if st.out=="audio" else "video")
    await c.message.edit_reply_markup(reply_markup=build_kb(st))
    await c.answer(f"Output: {st.out}")

@dp.callback_query(F.data.startswith("bg:"))
async def cb_bg(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    st.bgmix = not st.bgmix
    await c.message.edit_reply_markup(reply_markup=build_kb(st))
    await c.answer(f"BG mix: {'on' if st.bgmix else 'off'}")

@dp.callback_query(F.data.startswith("cmp:"))
async def cb_cmp(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    st.compress = not st.compress
    await c.message.edit_reply_markup(reply_markup=build_kb(st))
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
    await c.message.edit_reply_markup(reply_markup=build_kb(st))
    await c.answer(f"ASR: {st.asr_model}")

@dp.callback_query(F.data.startswith("lang:"))
async def cb_lang(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    st.lang = "auto" if st.lang == "en" else "en"
    await c.message.edit_reply_markup(reply_markup=build_kb(st))
    await c.answer(f"Lang: {st.lang}")

@dp.callback_query(F.data.startswith("clean:"))
async def cb_clean(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    st.clean_audio = not st.clean_audio
    await c.message.edit_reply_markup(reply_markup=build_kb(st))
    await c.answer(f"Clean audio: {'on' if st.clean_audio else 'off'}")

@dp.callback_query(F.data.startswith("job:cancel"))
async def cb_cancel(c: CallbackQuery):
    st = SESSIONS.setdefault(c.from_user.id, SessionState())
    if st.running_task and not st.running_task.done():
        st.running_task.cancel()
        await c.answer("Canceled.")
    else:
        await c.answer("No job running.")

# ==================== VIDEO HANDLER ====================
@dp.message(F.video | F.document)
async def on_video(m: Message):
    if not allowed_user(m.from_user.id):
        await m.reply("🚫 Access denied. This bot is in admin-only mode.")
        return

    st = SESSIONS.setdefault(m.from_user.id, SessionState())

    # Duration guard
    vid_dur = 0
    if m.video:
        vid_dur = m.video.duration or 0
    if vid_dur and vid_dur > MAX_DURATION_MIN * 60:
        await m.reply(f"Video too long. Max {MAX_DURATION_MIN} minutes (set MAX_DURATION_MIN in .env).")
        return

    if st.running_task and not st.running_task.done():
        await m.reply("You already have a running job. Use /cancel to stop it.")
        return

    work = user_workdir(m.from_user.id)
    ensure_dir(work)
    in_path = os.path.join(work, "input.mp4")

    await update_status(m, st, "⬇️ Downloading video…")
    file_obj = m.video or m.document
    file = await bot.get_file(file_obj.file_id)
    await bot.download_file(file.file_path, in_path)

    st.running_task = asyncio.create_task(process_job(m, st, in_path))
    try:
        await st.running_task
    except asyncio.CancelledError:
        await update_status(m, st, "❌ Job canceled.")
    finally:
        st.running_task = None

# ==================== CORE PIPELINE ====================
async def process_job(m: Message, st: SessionState, in_path: str):
    work = os.path.dirname(in_path)
    try:
        await update_status(m, st, "🎵 Extracting audio…")
        wav16k = os.path.join(work, "original.wav")
        await asyncio.to_thread(extract_audio_wav, in_path, wav16k, 16000, True)

        await update_status(m, st, f"🧠 Transcribing (ASR={st.asr_model}, lang={st.lang})…")
        segments = await asyncio.to_thread(transcribe_segments, wav16k, st.asr_model, st.lang)
        if not segments:
            await update_status(m, st, "I couldn't detect any speech.")
            return

        await update_status(m, st, "🌐 Translating to Hindi…")
        segments_hi = await asyncio.to_thread(translate_segments_en_hi, segments)

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

        await update_status(m, st, "🎛️ Post-processing audio…")
        await asyncio.to_thread(postprocess_audio_inplace, out_wav, st.clean_audio)

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
          
