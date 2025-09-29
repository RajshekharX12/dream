from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.types import InlineKeyboardMarkup

MODES = {
    "clone": "🎭 Clone Voice (XTTS)",
    "simple": "🗣️ Simple Hindi Voice",
}

ACTIONS = {
    "audio": "🎧 Send Audio Only",
    "video": "🎬 Replace Video Audio",
}

def start_kb() -> InlineKeyboardMarkup:
    kb = InlineKeyboardBuilder()
    kb.button(text=MODES["clone"], callback_data="mode:clone")
    kb.button(text=MODES["simple"], callback_data="mode:simple")
    kb.button(text=ACTIONS["video"], callback_data="action:video")
    kb.button(text=ACTIONS["audio"], callback_data="action:audio")
    kb.adjust(1)
    return kb.as_markup()
