        .output(vin.video, ain.audio, out_video, vcodec="copy", acodec="aac", strict="experimentDEVICE == "cpu":
        s(f"⚠️ Video is ~{dur_min:.1f} min on CPU — this can take a long time. Consider GPU (DEVICE=cuda).")

    s("🗣️ Transcribing English…")
    whisper = init_whisper()
    segments_iter, _info = whisper.transcribe(job.ref_audio_path, language="en")
    seg_times: List[Tuple[float, float]] = []
    seg_text_en: List[str] = []
    for seg in segments_iter:
        text = (getattr(seg, "text", "") or "").strip()
text:
            continue
        start = float(getattr(seg, "start", 0.0) or 0.0)
        end = float(getattr(seg, "end", 0.0) or 0.0)
        if end <= start:
            continue
        seg_times.append((start, end))
        seg_text_en.append(text)

    if not seg_times:
        raise RuntimeError("No usa

    if job.canceled:
        s("❌ Canceled.")
        return

    s("🌐 Translati
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
