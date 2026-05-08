from __future__ import annotations
from datetime import datetime
from pathlib import Path
from typing import Optional
import pandas as pd
from sqlalchemy.orm import Session
from app.models import SelectedSite, KpiData, PowerAlarm, ImportLog
from app.services.analytics import recalculate_site_alarm_averages, recalculate_revenue_priority, apply_alarm_states_from_dataframe
from app.services.ml_model import train_down_time_model, predict_for_imported_alarms

SITE_COLUMNS = {"site name": "site_name", "region": "region"}
KPI_COLUMNS = {
    "start time": "start_time",
    "period (min)": "period_min",
    "site name": "site_name",
    "data traffic volume (mb)": "data_traffic_mb",
    "voice traffic volume (erl)": "voice_traffic_erl",
}
ALARM_COLUMNS = {
    "incident id": "incident_id",
    "site name": "site_name",
    "region": "region",
    "event date": "event_date",
    "alarm sequence": "alarm_sequence",
    "alarm name": "alarm_name",
    "occurred on": "occurred_on",
    "cleared on": "cleared_on",
    "alarm duration (min)": "alarm_duration_min",
}

def _read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)

def _normalize_columns(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    normalized = {str(c).strip().lower(): c for c in df.columns}
    rename = {}
    missing = []
    for expected, new_name in mapping.items():
        if expected in normalized:
            rename[normalized[expected]] = new_name
        else:
            missing.append(expected)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    return df.rename(columns=rename)[list(mapping.values())]

def import_selected_sites(db: Session, file_path: str | Path, import_tag: str = "training") -> int:
    df = _normalize_columns(_read_table(file_path), SITE_COLUMNS).dropna(subset=["site_name"])
    count = 0
    for row in df.to_dict("records"):
        site = db.query(SelectedSite).filter(SelectedSite.site_name == str(row["site_name"]).strip()).first()
        if not site:
            site = SelectedSite(site_name=str(row["site_name"]).strip(), region=str(row.get("region", "")).strip())
            db.add(site)
        else:
            site.region = str(row.get("region", site.region)).strip()
        count += 1
    db.add(ImportLog(file_name=Path(file_path).name, import_type="selected_sites", import_tag=import_tag, row_count=count, message="Imported selected site list"))
    db.commit()
    return count

def import_kpi_data(db: Session, file_path: str | Path, import_tag: str = "training", recalculate: bool = True) -> int:
    df = _normalize_columns(_read_table(file_path), KPI_COLUMNS).dropna(subset=["start_time", "site_name"])
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["period_min"] = pd.to_numeric(df["period_min"], errors="coerce").fillna(60).astype(int)
    df["data_traffic_mb"] = pd.to_numeric(df["data_traffic_mb"], errors="coerce").fillna(0.0)
    df["voice_traffic_erl"] = pd.to_numeric(df["voice_traffic_erl"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["start_time"])
    count = 0
    batch = []
    for row in df.to_dict("records"):
        st = row["start_time"].to_pydatetime() if hasattr(row["start_time"], "to_pydatetime") else row["start_time"]
        batch.append(KpiData(
            start_time=st,
            period_min=int(row["period_min"]),
            site_name=str(row["site_name"]).strip(),
            data_traffic_mb=float(row["data_traffic_mb"]),
            voice_traffic_erl=float(row["voice_traffic_erl"]),
            hour_of_day=int(st.hour),
            import_tag=import_tag,
        ))
        count += 1
        if len(batch) >= 1000:
            db.bulk_save_objects(batch)
            db.commit()
            batch = []
    if batch:
        db.bulk_save_objects(batch)
    db.add(ImportLog(file_name=Path(file_path).name, import_type="kpi_data", import_tag=import_tag, row_count=count, message="Imported KPI records"))
    db.commit()
    if recalculate:
        recalculate_revenue_priority(db)
    return count

def import_power_alarms(db: Session, file_path: str | Path, import_tag: str = "training", is_training: bool = True) -> int:
    df = _normalize_columns(_read_table(file_path), ALARM_COLUMNS).dropna(subset=["incident_id", "site_name", "alarm_name", "occurred_on"])
    df["occurred_on"] = pd.to_datetime(df["occurred_on"], errors="coerce")
    df["cleared_on"] = pd.to_datetime(df["cleared_on"], errors="coerce")
    df["event_date"] = pd.to_datetime(df["event_date"], errors="coerce").dt.date
    df["alarm_sequence"] = pd.to_numeric(df["alarm_sequence"], errors="coerce").fillna(0).astype(int)
    df["alarm_duration_min"] = pd.to_numeric(df["alarm_duration_min"], errors="coerce").fillna(0.0)
    df = df.dropna(subset=["occurred_on"])
    count = 0
    batch = []
    for row in df.to_dict("records"):
        occurred = row["occurred_on"].to_pydatetime() if hasattr(row["occurred_on"], "to_pydatetime") else row["occurred_on"]
        cleared = None if pd.isna(row["cleared_on"]) else (row["cleared_on"].to_pydatetime() if hasattr(row["cleared_on"], "to_pydatetime") else row["cleared_on"])
        batch.append(PowerAlarm(
            incident_id=str(row["incident_id"]).strip(),
            site_name=str(row["site_name"]).strip(),
            region=str(row.get("region", "")).strip(),
            event_date=row.get("event_date"),
            alarm_sequence=int(row["alarm_sequence"]),
            alarm_name=str(row["alarm_name"]).strip(),
            occurred_on=occurred,
            cleared_on=cleared,
            alarm_duration_min=float(row["alarm_duration_min"]),
            import_tag=import_tag,
        ))
        count += 1
        if len(batch) >= 1000:
            db.bulk_save_objects(batch)
            db.commit()
            batch = []
    if batch:
        db.bulk_save_objects(batch)
    db.add(ImportLog(file_name=Path(file_path).name, import_type="power_alarms", import_tag=import_tag, row_count=count, message="Imported power alarm records"))
    db.commit()
    recalculate_site_alarm_averages(db)
    train_down_time_model(db)
    if not is_training:
        apply_alarm_states_from_dataframe(db, df)
        predict_for_imported_alarms(db, df)
    return count
