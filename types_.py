from dataclasses import dataclass

@dataclass
class SessionState:
    mode: str = "clone"   # or "simple"
    action: str = "video" # or "audio"
