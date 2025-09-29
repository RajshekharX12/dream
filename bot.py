import asyncio
import os
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, CallbackQuery, FSInputFile
from aiogram.utils.chat_action import ChatActionSender

from config import BOT_TOKEN, MAX_DURATION_MIN
from keyboards import start_kb
from types_ import SessionState
from files_utils import new_workdir
from logging_utils import logger
from stt import transcribe
from translate_hi import build_translator, translate_segments
from voice_clone import VoiceCloner
from mux import compose_dubbed_track, replace_video_audio
from ffmpeg_utils import extract_audio_wav, ffmpeg_extract

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()
sessions: dict[int, SessionState] = {}

@dp.message(CommandStart())
async def cmd_start(m: Message):
    sessions[m.from_user.id] = SessionState()
    await m.answer(
        "Send me an English video and I’ll return a **Hindi dub**.

"
        "Use the buttons to select **Clone Voice** or **Simple Hindi**, and choose **video** or **audio** output.",
        reply_markup=start_kb(),
        parse_mode="Markdown",
    )

@dp.callback_query(F.data.startswith("mode:"))
async def set_mode(c: CallbackQuery):
    mode = c.data.split(":", 1)[1]
    st = sessions.setdefault(c.from_user.id, SessionState())
    st.mode = mode
    await c.answer(f"Mode set: {mode}")

@dp.callback_query(F.data.startswith("action:"))
async def set_action(c: CallbackQuery):
    action = c.data.split(":", 1)[1]
    st = sessions.setdefault(c.from_user.id, SessionState())
    st.action = action
    await c.answer(f"Output: {action}")

@dp.message(F.video | F.document)
async def handle_video(m: Message):
    user_id = m.from_user.id
    st = sessions.setdefault(user_id, SessionState())

    file_obj = None
    duration = 0
    if m.video:
        file_obj = m.video
        duration = m.video.duration or 0
    elif m.document and (m.document.mime_type or "").startswith("video"):
        file_obj = m.document

    if not file_obj:
        await m.reply("Please send a video (mp4/mov).")
        return

    if duration and duration > MAX_DURATION_MIN * 60:
        await m.reply(f"Video too long. Max {MAX_DURATION_MIN} minutes.")
        return

    workdir = new_workdir(user_id)
    in_path = os.path.join(workdir, "input.mp4")

    await m.answer("Downloading video…")
    # aiogram v3: bot.download accepts File, file_id, or attachment object
    await bot.download(file=file_obj, destination=in_path)

    await m.answer("Extracting audio…")
    wav16k = extract_audio_wav(in_path, os.path.join(workdir, "original.wav"), sr=16000)

    await m.answer("Transcribing (English)…")
    async with ChatActionSender(bot=bot, chat_id=m.chat.id, action="typing"):
        segments = await asyncio.to_thread(transcribe, wav16k)
    if not segments:
        await m.reply("I couldn’t detect any English speech.")
        return

    await m.answer("Translating to Hindi…")
    translator = await asyncio.to_thread(build_translator)
    segments_hi = await asyncio.to_thread(translate_segments, segments, translator)

    # Reference clip: first segment (6–8s)
    ref_path = os.path.join(workdir, "ref.wav")
    s0 = segments[0]
    ref_start = max(0.0, float(s0["start"]))
    ref_dur = max(6.0, min(8.0, float(s0["end"]) - ref_start))
    ffmpeg_extract(wav16k, ref_path, ref_start, ref_start + ref_dur)

    if st.mode == "clone":
        await m.answer("Cloning voice & synthesizing Hindi… (XTTS)")
        cloner = VoiceCloner()
    else:
        await m.answer("Synthesizing Hindi voice… (simple)")
        class _Simple(VoiceCloner):
            def synthesize(self, text_hi: str, ref_wav: str):
                wav = self.tts.tts(text=text_hi, speaker=None, language="hi")
                import numpy as np
                import numpy as _np
                w = _np.asarray(wav, dtype=float)
                return w.astype("float32")
        cloner = _Simple()

    dubbed = os.path.join(workdir, "dubbed.wav")
    await m.answer("Composing dubbed track…")
    dubbed = await asyncio.to_thread(
        compose_dubbed_track,
        wav16k,
        segments_hi,
        ref_path,
        dubbed,
        cloner,
    )

    if st.action == "audio":
        await m.answer_document(document=FSInputFile(dubbed), caption="Hindi dub (audio)")
        return

    await m.answer("Muxing back into video…")
    out_video = os.path.join(workdir, "dubbed.mp4")
    await asyncio.to_thread(replace_video_audio, in_path, dubbed, out_video)
    await m.answer_video(video=FSInputFile(out_video), caption="Hindi dub (cloned voice)")


def main() -> None:
    logger.info("Starting bot…")
    dp.run_polling(bot)

if __name__ == "__main__":
    main()
