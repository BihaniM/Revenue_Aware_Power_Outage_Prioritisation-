from pathlib import Path
import argparse
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.database import init_db, SessionLocal
from app.services.importer import import_kpi_data, import_power_alarms, import_selected_sites

parser = argparse.ArgumentParser(description="Import selected-site, KPI or power-alarm data into the Telco Priority app database")
parser.add_argument("--type", choices=["sites", "kpi", "alarm"], required=True)
parser.add_argument("--file", required=True)
parser.add_argument("--training", action="store_true")
args = parser.parse_args()

init_db()
db = SessionLocal()
try:
    if args.type == "sites":
        count = import_selected_sites(db, args.file, import_tag="cli")
    elif args.type == "kpi":
        count = import_kpi_data(db, args.file, import_tag="cli")
    else:
        count = import_power_alarms(db, args.file, import_tag="cli", is_training=args.training)
    print(f"Imported {count} rows")
finally:
    db.close()
