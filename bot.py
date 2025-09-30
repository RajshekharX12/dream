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
