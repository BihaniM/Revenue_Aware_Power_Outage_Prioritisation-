from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.database import init_db, SessionLocal
from app.services.ml_model import train_down_time_model

init_db()
db = SessionLocal()
try:
    ok = train_down_time_model(db)
    print("Model trained successfully." if ok else "Not enough training records to train the model.")
finally:
    db.close()
