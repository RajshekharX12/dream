import os
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.environ.get("BOT_TOKEN")
USE_CUDA = os.environ.get("USE_CUDA", "true").lower() == "true"
MAX_DURATION_MIN = int(os.environ.get("MAX_DURATION_MIN", 20))
ASR_MODEL = os.environ.get("ASR_MODEL", "large-v3")

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN is not set in environment/.env")
