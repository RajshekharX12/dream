# bot.py
# Rose-lite single-file moderation bot (aiogram v3)
# Features:
# - Commandless moderation: ban/unban/mute/unmute/kick/warn (reply + word)
# - Rich inline "Manage User" menu (reply 'menu' / 'manage' to open)
# - Clean service join/left (ON by default) + welcome text
# - Antilink (invite links), soft antiflood
# - Locks (stickers/gifs/photos/links/voice/video)
# - Purge N / purge user, Pin / Pin silent / Unpin
# - Notes (save/get/list/del), Rules (set/show)
# - Info (id, whois), Adminlist, toggles (clean/antilink/flood)
# - Police: system snapshot + natural sentence via SafoneAPI (graceful fallback)
# Only env: BOT_TOKEN

import asyncio
import logging
import os
import re
import time
import json
from html import escape as h
from datetime import datetime, timedelta, timezone
from shutil import disk_usage

from aiogram import Bot, Dispatcher, Router, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode, ChatType
from aiogram.exceptions import TelegramBadRequest, TelegramForbiddenError
from aiogram.types import (
    Message,
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    ChatPermissions,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

try:
    import aiohttp  # aiogram depends on aiohttp; should be available after installing aiogram
except Exception:  # pragma: no cover
    aiohttp = None  # handled gracefully in police()

# -----------------------
# Config & Logging
# -----------------------
BOT_TOKEN = os.getenv("8287015753:AAGoGYF_u6-OqfrqGF1_xPY8yIW5FiD9MtE", "").strip()
if not BOT_TOKEN:
    raise SystemExit("Please set BOT_TOKEN:  export BOT_TOKEN=123456:ABCDEF")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
log = logging.getLogger("rose-lite-onefile")

# -----------------------
# Globals (in-memory store; no DB by design)
# -----------------------
GROUPS = {ChatType.GROUP, ChatType.SUPERGROUP}
ADMIN_STATUSES = {"creator", "administrator"}

# Per-chat settings (defaults populated lazily)
SETTINGS: dict[int, dict] = {}
# Warn counters
WARNS: dict[tuple[int, int], int] = {}
# Notes per chat: {chat_id: {key: value}}
NOTES: dict[int, dict[str, str]] = {}
# Rules per chat
RULES: dict[int, str] = {}
# Locks per chat: {chat_id: { "stickers": bool, ... }}
LOCKS: dict[int, dict[str, bool]] = {}
# Pending interactive actions (admin_id -> payload)
PENDING: dict[int, dict] = {}
# Flood window
FLOOD_BUCKET: dict[tuple[int, int], list[float]] = {}

# Permissions
NO_SPEAK = ChatPermissions(can_send_messages=False)
YES_SPEAK = ChatPermissions(
    can_send_messages=True,
    can_send_audios=True,
    can_send_documents=True,
    can_send_photos=True,
    can_send_videos=True,
    can_send_video_notes=True,
    can_send_voice_notes=True,
    can_send_polls=True,
    can_send_other_messages=True,
    can_add_web_page_previews=True,
)

# Regex helpers
DUR_RE = re.compile(r"\b(\d+)\s*([smhdw])\b", re.I)
PERM_RE = re.compile(r"\b(perm|permanent|forever)\b", re.I)
INVITE_RE = re.compile(r"(?:t\.me\/joinchat\/|t\.me\/\+|telegram\.me\/\+)", re.I)

# -----------------------
# Utilities
# -----------------------
def default_settings(chat_id: int) -> dict:
    st = SETTINGS.get(chat_id)
    if st is None:
        st = {
            "clean_service": 1,  # ON
            "welcome_text": "👋 Welcome {mention}! Please be respectful.",
            "antilink": 1,
            "flood_limit": 7,
            "flood_interval": 5,  # seconds
        }
        SETTINGS[chat_id] = st
    return st

def in_group(m: Message) -> bool:
    return getattr(m.chat, "type", None) in GROUPS

async def is_admin(bot: Bot, chat_id: int, user_id: int) -> bool:
    try:
        mem = await bot.get_chat_member(chat_id, user_id)
        return (getattr(mem, "status", None) in ADMIN_STATUSES) or getattr(mem, "is_anonymous", False)
    except TelegramBadRequest:
        return False

def mention(u) -> str:
    name = h(getattr(u, "full_name", None) or getattr(u, "first_name", "user"))
    return f'<a href="tg://user?id={u.id}">{name}</a>'

async def autodel(msg: Message, sec: float = 5.0):
    try:
        await asyncio.sleep(sec)
        await msg.delete()
    except Exception:
        pass

def parse_duration_utc(text: str, *, default_minutes: int | None):
    """Return (until_date, is_perm). >366 days == permanent in Telegram.
       If no duration and default_minutes is None => permanent."""
    if PERM_RE.search(text or ""):
        return datetime.now(timezone.utc) + timedelta(days=400), True
    m = DUR_RE.search(text or "")
    if not m:
        if default_minutes is None:
            return datetime.now(timezone.utc) + timedelta(days=400), True
        return datetime.now(timezone.utc) + timedelta(minutes=default_minutes), False
    n = int(m.group(1))
    unit = m.group(2).lower()
    delta = {"s": "seconds", "m": "minutes", "h": "hours", "d": "days", "w": "weeks"}[unit]
    return datetime.now(timezone.utc) + timedelta(**{delta: n}), False

def fmt_dur(until, is_perm: bool) -> str:
    if is_perm or until is None:
        return "permanent"
    return f"until <code>{until:%Y-%m-%d %H:%M UTC}</code>"

def strip_reason(text: str, keyword: str) -> str:
    if not text:
        return ""
    rest = re.sub(fr"(?is)^\s*{re.escape(keyword)}\b", "", text).strip()
    rest = DUR_RE.sub("", rest, count=1)
    rest = PERM_RE.sub("", rest)
    return rest.strip()

# CPU/RAM helpers (stdlib only; no psutil)
def _cpu_percent_sample(interval: float = 0.2) -> float:
    def read():
        with open("/proc/stat", "r") as f:
            line = f.readline()
        parts = [int(x) for x in line.strip().split()[1:8]]  # user,nice,system,idle,iowait,irq,softirq
        idle = parts[3] + parts[4]
        total = sum(parts)
        return idle, total
    try:
        idle1, total1 = read()
        time.sleep(interval)
        idle2, total2 = read()
        idle_delta = idle2 - idle1
        total_delta = total2 - total1
        if total_delta <= 0:
            return 0.0
        return round(100.0 * (1.0 - (idle_delta / total_delta)), 1)
    except Exception:
        return 0.0

def _mem_info_mb():
    try:
        with open("/proc/meminfo", "r") as f:
            data = f.read()
        def val(k):
            m = re.search(rf"^{k}:\s+(\d+)\s+kB", data, re.M)
            return int(m.group(1)) if m else 0
        total_kb = val("MemTotal")
        avail_kb = val("MemAvailable")
        used_kb = total_kb - avail_kb
        return used_kb // 1024, total_kb // 1024
    except Exception:
        return 0, 0

def _uptime_hms():
    try:
        with open("/proc/uptime", "r") as f:
            s = float(f.readline().split()[0])
        d, rem = divmod(int(s), 86400)
        h, rem = divmod(rem, 3600)
        m, _ = divmod(rem, 60)
        if d:
            return f"{d}d {h}h {m}m"
        return f"{h}h {m}m"
    except Exception:
        return "n/a"

# -----------------------
# Routers & Bot
# -----------------------
bot = Bot(BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
dp = Dispatcher()
r = Router(name="core")
dp.include_router(r)

# -----------------------
# Welcome & clean service
# -----------------------
@r.message(F.new_chat_members)
async def welcome_and_clean(m: Message):
    if not in_group(m):
        return
    st = default_settings(m.chat.id)
    for u in m.new_chat_members:
        txt = st["welcome_text"].replace("{mention}", mention(u))
        await m.answer(txt)
    if st.get("clean_service"):
        try:
            await m.delete()
        except Exception:
            pass

@r.message(F.left_chat_member)
async def clean_left(m: Message):
    if not in_group(m):
        return
    st = default_settings(m.chat.id)
    if st.get("clean_service"):
        try:
            await m.delete()
        except Exception:
            pass

# -----------------------
# Antispam / Flood / Locks enforcement
# -----------------------
LOCKABLE = {"stickers", "gifs", "photos", "links", "voice", "video"}

@r.message()
async def antispam_and_locks(m: Message):
    if not in_group(m):
        return

    st = default_settings(m.chat.id)

    # Skip admins for antispam enforcement
    if await is_admin(bot, m.chat.id, m.from_user.id):
        return

    # Locks
    lcfg = LOCKS.get(m.chat.id) or {}
    if lcfg.get("stickers") and m.sticker:
        return await _safe_del(m)
    if lcfg.get("gifs") and (m.animation or (m.document and getattr(m.document, "mime_type", "").startswith("video/"))):
        return await _safe_del(m)
    if lcfg.get("photos") and m.photo:
        return await _safe_del(m)
    if lcfg.get("voice") and (m.voice or m.audio):
        return await _safe_del(m)
    if lcfg.get("video") and (m.video or m.video_note):
        return await _safe_del(m)
    if lcfg.get("links") and m.text and ("http://" in m.text or "https://" in m.text):
        return await _safe_del(m)

    # Antilink (join invites)
    if st.get("antilink") and m.text and INVITE_RE.search(m.text):
        return await _safe_del(m)

    # Soft flood (window count)
    iv = int(st.get("flood_interval", 5))
    lim = int(st.get("flood_limit", 7))
    key = (m.chat.id, m.from_user.id)
    now = time.time()
    times = FLOOD_BUCKET.get(key) or []
    times.append(now)
    FLOOD_BUCKET[key] = [t for t in times if now - t <= iv]
    if len(FLOOD_BUCKET[key]) > lim:
        await _safe_del(m)
        try:
            warn = await m.answer(f"🤖 Slow down {mention(m.from_user)} — flooding detected.")
            asyncio.create_task(autodel(warn, 4))
        except Exception:
            pass

async def _safe_del(m: Message):
    try:
        await m.delete()
    except Exception:
        pass

# -----------------------
# Moderation (commandless text)
# -----------------------
BASE = F.text & F.reply_to_message

def _audit(action: str, mod, target, duration: str, reason: str, case_id: str) -> str:
    if action.lower() == "ban" and not reason:
        reason = "idk why fuck banned him"
    reason = reason or "—"
    return (
        f"🛡️ <b>{h(action.upper())}</b>\n"
        f"👮 By: {mention(mod)}\n"
        f"🎯 Target: {mention(target)}\n"
        f"⏱ Duration: {duration}\n"
        f"📝 Reason: {h(reason)}\n"
        f"🗂 Case: <code>{case_id}</code>"
    )

@r.message(BASE.filter(lambda m: re.match(r"(?is)^\s*ban\b", m.text or "")))
async def ban_text(m: Message):
    if not (in_group(m) and await is_admin(bot, m.chat.id, m.from_user.id)):
        return
    target = m.reply_to_message.from_user
    until, is_perm = parse_duration_utc(m.text, default_minutes=None)
    dur_txt = fmt_dur(None if is_perm else until, is_perm)
    reason = strip_reason(m.text, "ban")
    case_id = f"BAN-{m.chat.id}-{m.message_id}"
    try:
        await bot.ban_chat_member(m.chat.id, target.id, until_date=None if is_perm else until)
        out = await m.reply(_audit("ban", m.from_user, target, dur_txt, reason, case_id))
        asyncio.create_task(autodel(m, 3)); asyncio.create_task(autodel(out, 9))
    except (TelegramForbiddenError, TelegramBadRequest) as e:
        await m.reply(f"❌ Ban failed: <code>{h(str(e))}</code>")

@r.message(BASE.filter(lambda m: re.match(r"(?is)^\s*unban\b", m.text or "")))
async def unban_text(m: Message):
    if not (in_group(m) and await is_admin(bot, m.chat.id, m.from_user.id)):
        return
    target = m.reply_to_message.from_user
    reason = strip_reason(m.text, "unban")
    case_id = f"UNBAN-{m.chat.id}-{m.message_id}"
    try:
        await bot.unban_chat_member(m.chat.id, target.id, only_if_banned=True)
        out = await m.reply(_audit("unban", m.from_user, target, "n/a", reason, case_id))
        asyncio.create_task(autodel(m, 3)); asyncio.create_task(autodel(out, 6))
    except (TelegramForbiddenError, TelegramBadRequest) as e:
        await m.reply(f"❌ Unban failed: <code>{h(str(e))}</code>")

@r.message(BASE.filter(lambda m: re.match(r"(?is)^\s*mute\b", m.text or "")))
async def mute_text(m: Message):
    if not (in_group(m) and await is_admin(bot, m.chat.id, m.from_user.id)):
        return
    target = m.reply_to_message.from_user
    until, is_perm = parse_duration_utc(m.text, default_minutes=60)
    reason = strip_reason(m.text, "mute")
    case_id = f"MUTE-{m.chat.id}-{m.message_id}"
    try:
        await bot.restrict_chat_member(m.chat.id, target.id, permissions=NO_SPEAK, until_date=None if is_perm else until)
        out = await m.reply(_audit("mute", m.from_user, target, fmt_dur(None if is_perm else until, is_perm), reason, case_id))
        asyncio.create_task(autodel(m, 3)); asyncio.create_task(autodel(out, 7))
    except (TelegramForbiddenError, TelegramBadRequest) as e:
        await m.reply(f"❌ Mute failed: <code>{h(str(e))}</code>")

@r.message(BASE.filter(lambda m: re.match(r"(?is)^\s*unmute\b", m.text or "")))
async def unmute_text(m: Message):
    if not (in_group(m) and await is_admin(bot, m.chat.id, m.from_user.id)):
        return
    target = m.reply_to_message.from_user
    reason = strip_reason(m.text, "unmute")
    case_id = f"UNMUTE-{m.chat.id}-{m.message_id}"
    try:
        await bot.restrict_chat_member(m.chat.id, target.id, permissions=YES_SPEAK)
        out = await m.reply(_audit("unmute", m.from_user, target, "n/a", reason, case_id))
        asyncio.create_task(autodel(m, 3)); asyncio.create_task(autodel(out, 6))
    except (TelegramForbiddenError, TelegramBadRequest) as e:
        await m.reply(f"❌ Unmute failed: <code>{h(str(e))}</code>")

@r.message(BASE.filter(lambda m: re.match(r"(?is)^\s*kick\b", m.text or "")))
async def kick_text(m: Message):
    if not (in_group(m) and await is_admin(bot, m.chat.id, m.from_user.id)):
        return
    target = m.reply_to_message.from_user
    reason = strip_reason(m.text, "kick")
    case_id = f"KICK-{m.chat.id}-{m.message_id}"
    try:
        await bot.ban_chat_member(m.chat.id, target.id)
        await asyncio.sleep(1.0)
        await bot.unban_chat_member(m.chat.id, target.id, only_if_banned=True)
        out = await m.reply(_audit("kick", m.from_user, target, "n/a", reason, case_id))
        asyncio.create_task(autodel(m, 3)); asyncio.create_task(autodel(out, 6))
    except (TelegramForbiddenError, TelegramBadRequest) as e:
        await m.reply(f"❌ Kick failed: <code>{h(str(e))}</code>")

@r.message(BASE.filter(lambda m: re.match(r"(?is)^\s*warn\b", m.text or "")))
async def warn_text(m: Message):
    if not (in_group(m) and await is_admin(bot, m.chat.id, m.from_user.id)):
        return
    target = m.reply_to_message.from_user
    reason = strip_reason(m.text, "warn")
    key = (m.chat.id, target.id)
    count = WARNS.get(key, 0) + 1
    WARNS[key] = count
    if count >= 3:
        until = datetime.now(timezone.utc) + timedelta(minutes=10)
        try:
            await bot.restrict_chat_member(m.chat.id, target.id, permissions=NO_SPEAK, until_date=until)
            WARNS[key] = 0
            out = await m.reply(_audit("warn → auto-mute", m.from_user, target, f"until <code>{until:%Y-%m-%d %H:%M UTC}</code>", (reason or "—")+" (3 warns)", f"WARN-{m.chat.id}-{m.message_id}"))
            asyncio.create_task(autodel(m, 3)); asyncio.create_task(autodel(out, 8))
        except (TelegramForbiddenError, TelegramBadRequest) as e:
            await m.reply(f"❌ Auto-mute failed: <code>{h(str(e))}</code>")
    else:
        out = await m.reply(f"⚠️ Warning {count}/3 for {mention(target)}\n👮 By: {mention(m.from_user)}\n📝 Reason: {h(reason or '—')}")
        asyncio.create_task(autodel(m, 3)); asyncio.create_task(autodel(out, 6))

# -----------------------
# Manage menu (inline UI)
# -----------------------
def manage_menu_kb(target_id: int) -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text="🔨 Ban perm", callback_data=f"b:{target_id}:perm")
    kb.button(text="🔨 Ban 1d",   callback_data=f"b:{target_id}:1d")
    kb.button(text="🔨 Ban 1h",   callback_data=f"b:{target_id}:1h")
    kb.button(text="🤐 Mute 10m", callback_data=f"m:{target_id}:10m")
    kb.button(text="🤐 Mute 1h",  callback_data=f"m:{target_id}:1h")
    kb.button(text="🤐 Mute perm",callback_data=f"m:{target_id}:perm")
    kb.button(text="🗣 Unmute",   callback_data=f"u:{target_id}:0")
    kb.button(text="↩️ Unban",    callback_data=f"ub:{target_id}:0")
    kb.button(text="🥾 Kick",     callback_data=f"k:{target_id}:0")
    kb.button(text="⚠️ Warn",     callback_data=f"w:{target_id}:0")
    kb.button(text="🧽 Purge user", callback_data=f"pu:{target_id}:0")
    kb.button(text="🗑 Delete msg",  callback_data=f"dm:{target_id}:0")
    kb.button(text="📌 Pin",         callback_data=f"pin:{target_id}:0")
    kb.button(text="🔕 Pin silent",  callback_data=f"pins:{target_id}:0")
    kb.button(text="📌 Unpin all",   callback_data=f"unpin:{target_id}:0")
    kb.button(text="🔒 Locks",       callback_data=f"locks:open:0")
    return kb.adjust(3,3,3,3,3).as_markup()

def locks_menu_kb(chat_id: int) -> InlineKeyboardMarkup:
    cfg = LOCKS.get(chat_id) or {}
    def badge(name):
        return "✅" if cfg.get(name) else "❌"
    kb = InlineKeyboardBuilder()
    for item in ("stickers","gifs","photos","links","voice","video"):
        kb.button(text=f"{badge(item)} {item}", callback_data=f"locks:toggle:{item}")
    kb.button(text="⬅️ Back", callback_data="locks:back")
    return kb.adjust(3,3,1).as_markup()

@r.message(F.reply_to_message & F.text.regexp(r"(?is)^\s*(menu|manage)\b"))
async def open_manage_menu(m: Message):
    if not (in_group(m) and await is_admin(bot, m.chat.id, m.from_user.id)):
        return
    target = m.reply_to_message.from_user
    await m.reply(f"🧰 <b>Manage</b> · {mention(target)}", reply_markup=manage_menu_kb(target.id))

@r.callback_query(F.data.regexp(r"^(b|m|u|ub|k|w|pu|dm|pin|pins|unpin):"))
async def on_manage_action(cq: CallbackQuery):
    if not cq.message or not cq.message.chat:
        return await cq.answer()
    chat = cq.message.chat
    if chat.type not in GROUPS:
        return await cq.answer("Groups only", show_alert=True)
    if not await is_admin(bot, chat.id, cq.from_user.id):
        return await cq.answer("Admins only", show_alert=True)

    try:
        parts = cq.data.split(":")
        act = parts[0]; uid = int(parts[1]); dur = parts[2]
    except Exception:
        return await cq.answer()

    # find a recent replied message target if needed
    target_id = uid
    action_text = ""
    try:
        if act == "b":  # ban
            until, is_perm = _dur_from_token(dur)
            await bot.ban_chat_member(chat.id, target_id, until_date=None if is_perm else until)
            action_text = f"🔨 Banned {target_id} ({'perm' if is_perm else dur})"
        elif act == "ub":
            await bot.unban_chat_member(chat.id, target_id, only_if_banned=True)
            action_text = f"↩️ Unbanned {target_id}"
        elif act == "m":
            until, is_perm = _dur_from_token(dur)
            await bot.restrict_chat_member(chat.id, target_id, permissions=NO_SPEAK, until_date=None if is_perm else until)
            action_text = f"🤐 Muted {target_id} ({'perm' if is_perm else dur})"
        elif act == "u":
            await bot.restrict_chat_member(chat.id, target_id, permissions=YES_SPEAK)
            action_text = f"🗣 Unmuted {target_id}"
        elif act == "k":
            await bot.ban_chat_member(chat.id, target_id)
            await asyncio.sleep(1.0)
            await bot.unban_chat_member(chat.id, target_id, only_if_banned=True)
            action_text = f"🥾 Kicked {target_id}"
        elif act == "w":
            key = (chat.id, target_id)
            WARNS[key] = WARNS.get(key, 0) + 1
            action_text = f"⚠️ Warned {target_id} ({WARNS[key]}/3)"
        elif act == "pu":
            # Best-effort purge recent user's messages (last ~250)
            ok = 0
            base = cq.message.message_id
            for mid in range(base-1, max(base-250, 1), -1):
                try:
                    msg = await bot.forw
