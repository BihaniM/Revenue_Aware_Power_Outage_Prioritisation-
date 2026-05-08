import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)

_load_env_file(BASE_DIR / ".env")

APP_NAME = os.getenv("APP_NAME", "Telco Revenue-Aware Power Outage Prioritizer")
APP_VERSION = os.getenv("APP_VERSION", "R1.1")
APP_PATCH = os.getenv("APP_PATCH", "P01-SUPABASE")
APP_BUILD = os.getenv("APP_BUILD", "2026-05-01")
APP_DEPLOYMENT_TARGET = os.getenv("APP_DEPLOYMENT_TARGET", "AWS EC2 + Apache + Supabase PostgreSQL")
APP_SECRET_KEY = os.getenv("APP_SECRET_KEY", "change-this-secret-key")
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR / 'data' / 'telco_priority.db'}")
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", BASE_DIR / "data" / "uploads"))
MODEL_DIR = Path(os.getenv("MODEL_DIR", BASE_DIR / "data" / "models"))
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
ENABLE_OLLAMA = os.getenv("ENABLE_OLLAMA", "false").lower() == "true"
DATA_CHARGE_PER_MB = float(os.getenv("DATA_CHARGE_PER_MB", "0.30"))
VOICE_CHARGE_PER_MIN = float(os.getenv("VOICE_CHARGE_PER_MIN", "1.50"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
