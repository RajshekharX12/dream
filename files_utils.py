import os
import uuid

TMP_DIR = os.path.join(os.getcwd(), "_botdata")
os.makedirs(TMP_DIR, exist_ok=True)

def new_workdir(user_id: int) -> str:
    d = os.path.join(TMP_DIR, f"u{user_id}_{uuid.uuid4().hex[:8]}")
    os.makedirs(d, exist_ok=True)
    return d
