from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.database import init_db, SessionLocal
from app.models import KpiData, PowerAlarm, SelectedSite, RevenuePriority, SiteAlarmAverage, Prediction, ImportLog
from app.services.importer import import_selected_sites, import_kpi_data, import_power_alarms
from app.services.analytics import recalculate_revenue_priority

SAMPLE = ROOT / "sample_data"

def main():
    init_db()
    db = SessionLocal()
    try:
        db.query(Prediction).delete()
        db.query(RevenuePriority).delete()
        db.query(SiteAlarmAverage).delete()
        db.query(KpiData).delete()
        db.query(PowerAlarm).delete()
        db.query(SelectedSite).delete()
        db.query(ImportLog).delete()
        db.commit()
        import_selected_sites(db, SAMPLE / "Selected_50_Sites.xlsx", import_tag="training")
        import_kpi_data(db, SAMPLE / "KPI-db.xlsx", import_tag="training", recalculate=False)
        import_power_alarms(db, SAMPLE / "Power-Alarm-db.xlsx", import_tag="training", is_training=True)
        recalculate_revenue_priority(db)
        print("Demo database seeded successfully.")
    finally:
        db.close()

if __name__ == "__main__":
    main()
