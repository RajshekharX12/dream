import os
import uuid
import asyncio
import shutil
import traceback
from dataclasses import dataclass, field
from typing import Optional, Dict, Callable, List

from telethon import TelegramClient, events, Button
from telethon.tl.types import DocumentAttributeVideo
from dotenv import load_dotenv

import ffmpeg
import numpy as np
import soundfile as sf
import librosa

from faster_whisper import WhisperModel
from transformers import pipeline

from TTS.api import TTS

# ----------------------- Config & Globals -----------------------

load_dotenv()
API_ID = int(os.getenv("API_ID", "0"))
API_HASH = os.getenv("API_HASH", "")
BOT_TOKEN = os.getenv("BOT_TOKEN", "")
DEVICE = os.getenv("DEVICE", "cpu").strip().lower()  # 'cpu' or 'cuda'
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")

WORKDIR = os.path.abspath("work")
os.makedirs(WORKDIR, exist_ok=True)

# Lazy-initialized heavy models
_whisper_model: Optional[WhisperModel] = None
_translator = None
_tts_model: Optional[TTS] = None

# In-memory job table
@dataclass
class Job:
    job_id: str
    user_id: int
    chat_id: int
    video_path: str
    audio_ref_path: str = ""
    dubbed_wav_path: str = ""
    out_video_path: str = ""
    canceled: bool = False
    status_cb: Optional[Callable[[str], None]] = None

JOBS: Dict[str, Job] = {}

# ----------------------- Utility -----------------------

def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found. Please install ffmpeg.")

def extract_audio_16k(video_path: str, out_wav: str) -> None:
    (
        ffmpeg
        .input(video_path)
        .output(out_wav, ac=1, ar=16000, vn=None, y=None)
        .overwrite_output()
        .run(quiet=True)
    )

def mux_audio(video_path: str, audio_path: str, out_video: str) -> None:
    video_in = ffmpeg.input(video_path)
    audio_in = ffmpeg.input(audio_path)
    (
        ffmpeg
        .output(video_in.video, audio_in.audio, out_video, vcodec="copy", acodec="aac", strict="experimental")
        .overwrite_output()
        .run(quiet=True)
    )

def time_stretch_to_match(src_wav: str, target_duration: float, out_wav: str) -> None:
    # Load audio with librosa and time-stretch to match target duration
    y, sr = librosa.load(src_wav, sr=None, mono=True)
    cur_dur = len(y) / sr
    if cur_dur <= 0.0:
        shutil.copyfile(src_wav, out_wav)
        return
    rate = max(0.5, min(2.0, target_duration / cur_dur))  # clamp speedup for quality
    y2 = librosa.effects.time_stretch(y, rate)
    sf.write(out_wav, y2, sr)

def get_video_duration(video_path: str) -> float:
    # Probe with ffprobe via ffmpeg-python
    probe = ffmpeg.probe(video_path)
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == "video":
            return float(stream.get("duration") or probe.get("format", {}).get("duration") or 0.0)
    return float(probe.get("format", {}).get("duration") or 0.0)

def get_wav_duration(path: str) -> float:
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate

def chunk_text(text: str, max_chars: int = 900) -> List[str]:
    # naive chunker on sentence-ish boundaries
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    out, cur = [], []
    total = 0
    for part in text.replace("\n", " ").split(". "):
        piece = (part + ". ").strip()
        if total + len(piece) > max_chars and cur:
            out.append(" ".join(cur).strip())
            cur, total = [], 0
        cur.append(piece)
        total += len(piece)
    if cur:
        out.append(" ".join(cur).strip())
    return out

def init_whisper() -> WhisperModel:
    global _whisper_model
    if _whisper_model is None:
        # compute_type int8 is RAM-friendly; set to "float32" for max quality if you have headroom
        _whisper_model = WhisperModel(WHISPER_MODEL_NAME, device=DEVICE, compute_type="int8")
    return _whisper_model

def init_translator():
    global _translator
    if _translator is None:
        # Helsinki-NLP/opus-mt-en-hi is small and reliable
        _translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-hi",
            tokenizer="Helsinki-NLP/opus-mt-en-hi"
        )
    return _translator

def init_tts() -> TTS:
    global _tts_model
    if _tts_model is None:
        # Cross-lingual voice cloning (speaker_wav) + Hindi language code "hi"
        _tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    return _tts_model

# ----------------------- Core Pipeline -----------------------

def pipeline_translate_dub(job: Job):
    """
    Runs in a thread (blocking). Uses job.status_cb for live updates.
    """
    s = job.status_cb or (lambda *_: None)
    ensure_ffmpeg()

    s("🎬 Extracting audio…")
    job.audio_ref_path = os.path.join(WORKDIR, f"{job.job_id}_orig.wav")
    extract_audio_16k(job.video_path, job.audio_ref_path)
    if job.canceled:
        s("❌ Canceled.")
        return

    s("🗣️ Transcribing English…")
    whisper = init_whisper()
    segments, info = whisper.transcribe(job.audio_ref_path, language="en")
    english_text = " ".join(seg.text.strip() for seg in segments if getattr(seg, "text", "").strip())
    if not english_text.strip():
        raise RuntimeError("No speech detected or transcription empty.")

    if job.canceled:
        s("❌ Canceled.")
        return

    s("🌐 Translating to Hindi…")
    translator = init_translator()
    chunks = chunk_text(english_text, max_chars=900)
    hi_parts = [translator(ch)[0]["translation_text"] for ch in chunks]
    hindi_text = " ".join(hi_parts).strip()
    if not hindi_text:
        raise RuntimeError("Translation failed (empty result).")

    if job.canceled:
        s("❌ Canceled.")
        return

    s("🧬 Cloning voice & synthesizing Hindi…")
    tts = init_tts()
    raw_tts_wav = os.path.join(WORKDIR, f"{job.job_id}_tts_raw.wav")
    # Use original audio as speaker reference to keep same voice
    tts.tts_to_file(
        text=hindi_text,
        speaker_wav=job.audio_ref_path,
        language="hi",
        file_path=raw_tts_wav,
    )

    if job.canceled:
        s("❌ Canceled.")
        return

    s("⏱️ Matching original timing…")
    target_dur = get_wav_duration(job.audio_ref_path)
    stretched_wav = os.path.join(WORKDIR, f"{job.job_id}_tts_sync.wav")
    time_stretch_to_match(raw_tts_wav, target_dur, stretched_wav)
    job.dubbed_wav_path = stretched_wav

    if job.canceled:
        s("❌ Canceled.")
        return

    s("🎧 Muxing new audio into video…")
    job.out_video_path = os.path.join(WORKDIR, f"{job.job_id}_dubbed.mp4")
    mux_audio(job.video_path, job.dubbed_wav_path, job.out_video_path)

    s("✅ Done! Sending the dubbed video…")

# ----------------------- Bot Setup -----------------------

client = TelegramClient("bot", API_ID, API_HASH)

def make_status_cb(message):
    loop = client.loop
    def _cb(text: str):
        asyncio.run_coroutine_threadsafe(message.edit(text), loop)
    return _cb

def is_video_message(msg) -> bool:
    if getattr(msg, "video", None):
        return True
    doc = getattr(msg, "document", None)
    if doc and getattr(doc, "mime_type", "") and doc.mime_type.startswith("video/"):
        return True
    # Some clients send MP4 as document without mime_type
    if doc and doc.attributes:
        return any(isinstance(a, DocumentAttributeVideo) for a in doc.attributes)
    return False

# ----------------------- Handlers -----------------------

@client.on(events.NewMessage(pattern=r"^/start"))
async def start_handler(event):
    await event.respond(
        "👋 Send me a short English video (mp4/mov). Then tap **Translate EN→HI (same voice)**.\n\n"
        "I’ll transcribe → translate → clone the voice → replace audio. "
        "For best results, use clear speech and ≤2–3 minutes.",
        buttons=[ [Button.inline("📜 Help", data=b"help")] ]
    )

@client.on(events.CallbackQuery(pattern=b"help"))
async def help_cb(event):
    await event.answer()
    await event.edit(
        "🛠️ **How it works**\n"
        "1) Send a video with clear English speech.\n"
        "2) Press **Translate EN→HI**.\n"
        "3) I’ll send a dubbed Hindi video using the **same voice**.\n\n"
        "Tip: For movie-grade lip sync, integrate Wav2Lip later. This bot already preserves duration to keep lips close."
    )

@client.on(events.NewMessage())
async def media_handler(event):
    msg = event.message
    if not is_video_message(msg):
        return  # ignore non-video messages silently

    # Save video
    job_id = uuid.uuid4().hex[:10]
    video_path = os.path.join(WORKDIR, f"{job_id}.mp4")
    await event.reply("⬇️ Downloading video…")
    await client.download_media(msg, file=video_path)

    # Store job
    job = Job(
        job_id=job_id,
        user_id=event.sender_id,
        chat_id=event.chat_id,
        video_path=video_path,
    )
    JOBS[job_id] = job

    # Ask for action
    await event.respond(
        f"🎥 Got your video.\nJob ID: `{job_id}`\nChoose an action:",
        buttons=[
            [Button.inline("🇮🇳 Translate EN→HI (same voice)", data=f"go|{job_id}".encode())],
            [Button.inline("🔄 Status", data=f"status|{job_id}".encode()),
             Button.inline("❌ Cancel", data=f"cancel|{job_id}".encode())],
        ]
    )

@client.on(events.CallbackQuery())
async def on_button(event):
    try:
        data = event.data.decode("utf-8")
    except Exception:
        await event.answer("Bad data.", alert=True)
        return

    if "|" not in data:
        return
    cmd, job_id = data.split("|", 1)
    job = JOBS.get(job_id)
    if not job:
        await event.answer("Job not found (maybe restarted).", alert=True)
        return

    if cmd == "status":
        state = (
            f"📄 Job `{job_id}`\n"
            f"• Video: `{os.path.basename(job.video_path)}`\n"
            f"• Canceled: {job.canceled}\n"
            f"• Dubbed: {'yes' if job.out_video_path else 'no'}"
        )
        await event.answer(state, alert=True)
        return

    if cmd == "cancel":
        job.canceled = True
        await event.edit(f"🚫 Job `{job_id}` canceled.")
        return

    if cmd == "go":
        # Create a live status message and run pipeline in thread
        status_msg = await event.respond(f"🧪 Starting job `{job_id}`…")
        job.status_cb = make_status_cb(status_msg)

        async def run_and_send():
            try:
                await asyncio.to_thread(pipeline_translate_dub, job)
                if job.canceled:
                    return
                if not job.out_video_path or not os.path.exists(job.out_video_path):
                    await status_msg.edit("❌ Failed to produce output.")
                    return
                await client.send_file(
                    job.chat_id,
                    job.out_video_path,
                    caption=f"✅ Dubbed (EN→HI, same voice)\nJob: `{job_id}`",
                    force_document=False,
                )
                await status_msg.delete()
            except Exception as e:
                tb = traceback.format_exc()
                await status_msg.edit(f"💥 Error:\n`{e}`\n\n```\n{tb}\n```")

        await event.edit(f"🏃 Processing job `{job_id}`…")
        asyncio.create_task(run_and_send())

# ----------------------- Main -----------------------

def main():
    if not API_ID or not API_HASH or not BOT_TOKEN:
        raise SystemExit("Please set API_ID, API_HASH, BOT_TOKEN in .env")
    print("Starting bot…")
    with client:
        client.start(bot_token=BOT_TOKEN)
        print("Bot is up. Press Ctrl+C to stop.")
        client.run_until_disconnected()

if __name__ == "__main__":
    main()
