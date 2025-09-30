import os
import uuid
import asyncio
import shutil
import traceback
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, Tuple

from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, F, Router
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.filters import CommandStart
from aiogram.utils.keyboard import InlineKeyboardBuilder

import ffmpeg
import numpy as np
import soundfile as sf
import librosa

from faster_whisper import WhisperModel
from transformers import pipeline
from TTS.api import TTS


# -------------------- Config --------------------

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
DEVICE = os.getenv("DEVICE", "cpu").strip().lower()   # 'cpu' or 'cuda'
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small").strip()

WORKDIR = os.path.abspath("work")
os.makedirs(WORKDIR, exist_ok=True)

if not BOT_TOKEN:
    raise SystemExit("BOT_TOKEN is not set in .env")

# Heavy models (lazy)
_whisper_model: Optional[WhisperModel] = None
_translator = None
_tts_model: Optional[TTS] = None

MAIN_LOOP: Optional[asyncio.AbstractEventLoop] = None


# -------------------- Utilities --------------------

def ensure_ffmpeg() -> None:
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        raise RuntimeError("ffmpeg/ffprobe not found. Install ffmpeg on the host OS.")

def extract_audio_16k(video_path: str, out_wav: str) -> None:
    (
        ffmpeg
        .input(video_path)
        .output(out_wav, ac=1, ar=16000, vn=None)
        .overwrite_output()
        .run(quiet=True)
    )

def mux_audio(video_path: str, audio_path: str, out_video: str) -> None:
    vin = ffmpeg.input(video_path)
    ain = ffmpeg.input(audio_path)
    (
        ffmpeg
        .output(vin.video, ain.audio, out_video, vcodec="copy", acodec="aac", strict="experimental")
        .overwrite_output()
        .run(quiet=True)
    )

def get_media_duration_sec(path: str) -> float:
    try:
        probe = ffmpeg.probe(path)
        dur = probe.get("format", {}).get("duration")
        return float(dur) if dur is not None else 0.0
    except Exception:
        return 0.0

def chunk_text_simple(text: str, max_chars: int = 500) -> List[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text]
    out, cur, total = [], [], 0
    for piece in text.replace("\n", " ").split(". "):
        piece = (piece + ". ").strip()
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
        _whisper_model = WhisperModel(WHISPER_MODEL_NAME, device=DEVICE, compute_type="int8")
    return _whisper_model

def init_translator():
    global _translator
    if _translator is None:
        _translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-hi",
            tokenizer="Helsinki-NLP/opus-mt-en-hi"
        )
    return _translator

def init_tts() -> TTS:
    global _tts_model
    if _tts_model is None:
        _tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")
    return _tts_model


# -------------------- Lip-sync aware synth --------------------

def synth_segment_to_duration(
    text_hi: str,
    speaker_wav: str,
    language: str,
    target_dur: float,
    sr_out: Optional[int] = None
) -> Tuple[np.ndarray, int]:
    """
    Synthesize Hindi speech in the same voice and time-stretch to target_dur.
    Returns (audio, sample_rate).
    """
    tts = init_tts()
    temp_path = speaker_wav + f".seg_{uuid.uuid4().hex}.wav"
    tts.tts_to_file(
        text=text_hi,
        speaker_wav=speaker_wav,
        language=language,
        file_path=temp_path,
    )
    y, sr = sf.read(temp_path, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)  # ensure mono
    if sr_out and sr != sr_out:
        y = librosa.resample(y, orig_sr=sr, target_sr=sr_out)
        sr = sr_out
    cur_dur = len(y) / sr if len(y) else 0.0
    if cur_dur > 0 and target_dur > 0:
        rate = max(0.5, min(2.0, target_dur / cur_dur))
        y = librosa.effects.time_stretch(y, rate)
    try:
        os.remove(temp_path)
    except Exception:
        pass
    return y.astype(np.float32), sr


def build_dubbed_track_from_segments(
    speaker_wav: str,
    seg_times: List[Tuple[float, float]],
    seg_text_hi: List[str]
) -> Tuple[str, int, float]:
    """
    Create a single WAV by synthesizing each translated segment and placing it at original timestamps.
    Returns (wav_path, sample_rate, total_duration).
    """
    sr_global: Optional[int] = None
    pieces: List[np.ndarray] = []
    cursor = 0.0

    for (start, end), text_hi in zip(seg_times, seg_text_hi):
        seg_target = max(0.01, end - start)
        gap = max(0.0, start - cursor)
        if sr_global is None:
            y_tmp, sr_tmp = synth_segment_to_duration(text_hi, speaker_wav, "hi", seg_target, None)
            sr_global = sr_tmp
            if gap > 0:
                pieces.append(np.zeros(int(gap * sr_global), dtype=np.float32))
            pieces.append(y_tmp)
        else:
            if gap > 0:
                pieces.append(np.zeros(int(gap * sr_global), dtype=np.float32))
            y_tmp, _ = synth_segment_to_duration(text_hi, speaker_wav, "hi", seg_target, sr_out=sr_global)
            pieces.append(y_tmp)
        cursor = end

    if sr_global is None:
        raise RuntimeError("No segments synthesized.")

    audio = np.concatenate(pieces) if pieces else np.zeros(1, dtype=np.float32)
    out_wav = os.path.join(WORKDIR, f"dub_{uuid.uuid4().hex}.wav")
    sf.write(out_wav, audio, sr_global)
    total_dur = len(audio) / sr_global
    return out_wav, sr_global, total_dur


# -------------------- Pipeline --------------------

@dataclass
class Job:
    job_id: str
    user_id: int
    chat_id: int
    video_path: str
    status_cb: Optional[Callable[[str], None]] = None
    canceled: bool = False
    ref_audio_path: str = ""
    out_audio_path: str = ""
    out_video_path: str = ""

JOBS: Dict[str, Job] = {}


def pipeline_translate_dub(job: Job):
    s = job.status_cb or (lambda *_: None)
    ensure_ffmpeg()

    s("🎬 Extracting reference audio…")
    job.ref_audio_path = os.path.join(WORKDIR, f"{job.job_id}_ref.wav")
    extract_audio_16k(job.video_path, job.ref_audio_path)
    if job.canceled:
        s("❌ Canceled.")
        return

    dur_min = get_media_duration_sec(job.video_path) / 60.0
    if dur_min > 15 and DEVICE == "cpu":
        s(f"⚠️ Video is ~{dur_min:.1f} min on CPU — this can take a long time. Consider GPU (DEVICE=cuda).")

    s("🗣️ Transcribing English…")
    whisper = init_whisper()
    segments_iter, _info = whisper.transcribe(job.ref_audio_path, language="en")
    seg_times: List[Tuple[float, float]] = []
    seg_text_en: List[str] = []
    for seg in segments_iter:
        text = (getattr(seg, "text", "") or "").strip()
        if not text:
            continue
        start = float(getattr(seg, "start", 0.0) or 0.0)
        end = float(getattr(seg, "end", 0.0) or 0.0)
        if end <= start:
            continue
        seg_times.append((start, end))
        seg_text_en.append(text)

    if not seg_times:
        raise RuntimeError("No usable speech segments found.")

    if job.canceled:
        s("❌ Canceled.")
        return

    s("🌐 Translating to Hindi…")
    translator = init_translator()
    seg_text_hi: List[str] = []
    for en in seg_text_en:
        chunks = chunk_text_simple(en, max_chars=400)
        hi_parts = [translator(ch)[0]["translation_text"] for ch in chunks]
        seg_text_hi.append(" ".join(hi_parts).strip())

    if job.canceled:
        s("❌ Canceled.")
        return

    s("🧬 Voice cloning & segment-aligned synthesis…")
    dubbed_wav, _sr, _dur = build_dubbed_track_from_segments(
        speaker_wav=job.ref_audio_path,
        seg_times=seg_times,
        seg_text_hi=seg_text_hi
    )
    job.out_audio_path = dubbed_wav

    if job.canceled:
        s("❌ Canceled.")
        return

    s("🎧 Muxing Hindi audio into original video…")
    out_path = os.path.join(WORKDIR, f"{job.job_id}_dubbed.mp4")
    mux_audio(job.video_path, job.out_audio_path, out_path)
    job.out_video_path = out_path

    s("✅ Done! Sending the dubbed video…")


# -------------------- Bot & Handlers --------------------

bot = Bot(token=BOT_TOKEN, parse_mode=None)
dp = Dispatcher()
router = Router()

def job_keyboard(job_id: str):
    kb = InlineKeyboardBuilder()
    kb.button(text="🇮🇳 Translate EN→HI (same voice)", callback_data=f"go|{job_id}")
    kb.button(text="🔄 Status", callback_data=f"status|{job_id}")
    kb.button(text="❌ Cancel", callback_data=f"cancel|{job_id}")
    kb.adjust(1, 2)
    return kb.as_markup()

@router.message(CommandStart())
async def on_start(message: Message):
    await message.answer(
        "👋 Send me an English video (mp4/mov). Tap **Translate EN→HI (same voice)**.\n\n"
        "I’ll transcribe → translate → clone the voice → replace audio.\n"
        "Tip: Clear speech gives best results. Long videos on CPU take a long time."
    )

@router.message()
async def on_video(message: Message):
    file_obj = None
    if message.video:
        file_obj = message.video
    elif message.document and message.document.mime_type and message.document.mime_type.startswith("video/"):
        file_obj = message.document
    else:
        return

    job_id = uuid.uuid4().hex[:10]
    video_path = os.path.join(WORKDIR, f"{job_id}.mp4")

    await message.reply("⬇️ Downloading video…")
    await bot.download(file_obj, destination=video_path)

    job = Job(
        job_id=job_id,
        user_id=message.from_user.id if message.from_user else 0,
        chat_id=message.chat.id,
        video_path=video_path,
    )
    JOBS[job_id] = job

    await message.answer(
        f"🎥 Got your video.\nJob ID: `{job_id}`\nChoose an action:",
        reply_markup=job_keyboard(job_id)
    )

@router.callback_query(F.data)
async def on_callback(cb: CallbackQuery):
    try:
        data = cb.data or ""
        if "|" not in data:
            await cb.answer()
            return
        cmd, job_id = data.split("|", 1)
        job = JOBS.get(job_id)
        if not job:
            await cb.answer("Job not found (maybe restarted).", show_alert=True)
            return

        if cmd == "status":
            state = (
                f"📄 Job `{job_id}`\n"
                f"• Canceled: {job.canceled}\n"
                f"• Output ready: {'yes' if job.out_video_path else 'no'}"
            )
            await cb.answer(state, show_alert=True)
            return

        if cmd == "cancel":
            job.canceled = True
            await cb.message.edit_text(f"🚫 Job `{job_id}` canceled.")
            await cb.answer()
            return

        if cmd == "go":
            status_msg = await cb.message.answer(f"🧪 Starting job `{job_id}`…")

            def status_cb(text: str):
                if MAIN_LOOP is not None:
                    asyncio.run_coroutine_threadsafe(status_msg.edit_text(text), MAIN_LOOP)

            job.status_cb = status_cb

            async def run_and_send():
                try:
                    await asyncio.to_thread(pipeline_translate_dub, job)
                    if job.canceled:
                        return
                    if not job.out_video_path or not os.path.exists(job.out_video_path):
                        await status_msg.edit_text("❌ Failed to produce output.")
                        return
                    await bot.send_video(
                        chat_id=job.chat_id,
                        video=FSInputFile(job.out_video_path),
                        caption=f"✅ Dubbed (EN→HI, same voice)\nJob: `{job_id}`"
                    )
                    await status_msg.delete()
                except Exception as e:
                    tb = traceback.format_exc()
                    await status_msg.edit_text(f"💥 Error:\n`{e}`\n\n```\n{tb}\n```")

            await cb.message.edit_text(f"🏃 Processing job `{job_id}`…")
            asyncio.create_task(run_and_send())
            await cb.answer()
            return

        await cb.answer()

    except Exception as e:
        try:
            await cb.answer(f"Error: {e}", show_alert=True)
        except Exception:
            pass


async def main():
    global MAIN_LOOP
    ensure_ffmpeg()
    MAIN_LOOP = asyncio.get_running_loop()
    dp.include_router(router)
    await dp.start_polling(bot)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
