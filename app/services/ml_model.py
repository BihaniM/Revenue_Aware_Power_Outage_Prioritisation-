from __future__ import annotations
from datetime import datetime
from pathlib import Path
import json
import joblib
import pandas as pd
from sqlalchemy.orm import Session
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from app.config import MODEL_DIR
from app.models import PowerAlarm, SiteAlarmAverage, RevenuePriority, Prediction
from app.services.analytics import ALARM_STATE_BY_SEQUENCE, STATE_WEIGHT

MODEL_PATH = MODEL_DIR / "down_time_model.joblib"
META_PATH = MODEL_DIR / "down_time_model_meta.json"

def _incident_training_frame(db: Session) -> pd.DataFrame:
    rows = db.query(PowerAlarm).all()
    if not rows:
        return pd.DataFrame()
    raw = pd.DataFrame([{
        "incident_id": r.incident_id,
        "site_name": r.site_name,
        "region": r.region,
        "alarm_name": r.alarm_name,
        "occurred_on": r.occurred_on,
        "cleared_on": r.cleared_on,
        "alarm_duration_min": r.alarm_duration_min,
    } for r in rows])
    records = []
    for (incident_id, site_name), g in raw.groupby(["incident_id", "site_name"]):
        by_name = {str(r.alarm_name).lower(): r for r in g.itertuples()}
        mains = by_name.get("mains input out of range")
        dc = by_name.get("power supply dc output out of range")
        down = by_name.get("enodeb out of service")
        if not mains or not dc or not down:
            continue
        mains_to_dc = (dc.occurred_on - mains.occurred_on).total_seconds() / 60
        dc_to_down = (down.occurred_on - dc.occurred_on).total_seconds() / 60
        site_down_duration = float(down.alarm_duration_min or 0)
        if site_down_duration == 0 and down.cleared_on:
            site_down_duration = (down.cleared_on - down.occurred_on).total_seconds() / 60
        if mains_to_dc < 0 or dc_to_down < 0 or site_down_duration < 0:
            continue
        records.append({
            "site_name": site_name,
            "region": mains.region or "Unknown",
            "hour_of_day": int(mains.occurred_on.hour),
            "day_of_week": int(mains.occurred_on.weekday()),
            "mains_to_dc_min": mains_to_dc,
            "dc_to_down_min": dc_to_down,
            "site_down_duration_min": site_down_duration,
        })
    return pd.DataFrame(records)

def train_down_time_model(db: Session) -> bool:
    df = _incident_training_frame(db)
    if len(df) < 10:
        return False
    X = df[["site_name", "region", "hour_of_day", "day_of_week"]]
    y = df[["mains_to_dc_min", "dc_to_down_min", "site_down_duration_min"]]
    model = Pipeline(steps=[
        ("preprocess", ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["site_name", "region"]),
            ("num", "passthrough", ["hour_of_day", "day_of_week"]),
        ])),
        ("regressor", RandomForestRegressor(n_estimators=120, random_state=42, min_samples_leaf=2, n_jobs=-1)),
    ])
    model.fit(X, y)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    META_PATH.write_text(json.dumps({
        "trained_at": datetime.utcnow().isoformat(),
        "training_records": int(len(df)),
        "target_columns": ["mains_to_dc_min", "dc_to_down_min", "site_down_duration_min"],
        "algorithm": "RandomForestRegressor with one-hot encoded site/region and time features",
    }, indent=2), encoding="utf-8")
    return True

def _fallback_prediction(db: Session, site_name: str) -> tuple[float, float, float]:
    avg = db.query(SiteAlarmAverage).filter(SiteAlarmAverage.site_name == site_name).first()
    if avg:
        return float(avg.avg_mains_to_dc_min or 45), float(avg.avg_dc_to_down_min or 35), float(avg.avg_site_down_duration_min or 90)
    all_avg = db.query(SiteAlarmAverage).all()
    if all_avg:
        return (
            sum(a.avg_mains_to_dc_min or 0 for a in all_avg) / len(all_avg),
            sum(a.avg_dc_to_down_min or 0 for a in all_avg) / len(all_avg),
            sum(a.avg_site_down_duration_min or 0 for a in all_avg) / len(all_avg),
        )
    return 45.0, 35.0, 90.0

def predict_duration(db: Session, site_name: str, region: str, event_time: datetime) -> tuple[float, float, float, str]:
    if MODEL_PATH.exists():
        try:
            model = joblib.load(MODEL_PATH)
            X = pd.DataFrame([{
                "site_name": site_name,
                "region": region or "Unknown",
                "hour_of_day": int(event_time.hour),
                "day_of_week": int(event_time.weekday()),
            }])
            pred = model.predict(X)[0]
            return float(max(pred[0], 0)), float(max(pred[1], 0)), float(max(pred[2], 0)), "ML model"
        except Exception:
            pass
    p = _fallback_prediction(db, site_name)
    return p[0], p[1], p[2], "site average fallback"

def predict_for_imported_alarms(db: Session, df: pd.DataFrame) -> int:
    db.query(Prediction).delete()
    if df.empty:
        db.commit()
        return 0
    work = df.copy()
    work["occurred_on"] = pd.to_datetime(work["occurred_on"], errors="coerce")
    work["alarm_sequence"] = pd.to_numeric(work["alarm_sequence"], errors="coerce").fillna(1).astype(int)
    latest = work.sort_values(["site_name", "occurred_on", "alarm_sequence"]).groupby("site_name").tail(1)
    count = 0
    for row in latest.to_dict("records"):
        event_time = row["occurred_on"]
        if hasattr(event_time, "to_pydatetime"):
            event_time = event_time.to_pydatetime()
        site_name = str(row["site_name"]).strip()
        region = str(row.get("region", "")).strip()
        state = ALARM_STATE_BY_SEQUENCE.get(int(row.get("alarm_sequence", 1)), "Power failure")
        mains_to_dc, dc_to_down, site_down_duration, _source = predict_duration(db, site_name, region, event_time)
        revenue = db.query(RevenuePriority).filter(RevenuePriority.site_name == site_name).first()
        data_rev = float(revenue.data_revenue_hour if revenue else 0.0)
        voice_rev = float(revenue.voice_revenue_hour if revenue else 0.0)
        total_rev = data_rev + voice_rev
        state_bonus = STATE_WEIGHT.get(state, 1) * 10000
        db.add(Prediction(
            site_name=site_name,
            region=region,
            alarm_state=state,
            event_time=event_time,
            predicted_mains_to_dc_min=round(mains_to_dc, 2),
            predicted_dc_to_down_min=round(dc_to_down, 2),
            predicted_site_down_duration_min=round(site_down_duration, 2),
            data_loss_1h=round(data_rev * 1, 2),
            data_loss_2h=round(data_rev * 2, 2),
            data_loss_3h=round(data_rev * 3, 2),
            data_loss_4h=round(data_rev * 4, 2),
            voice_loss_1h=round(voice_rev * 1, 2),
            voice_loss_2h=round(voice_rev * 2, 2),
            voice_loss_3h=round(voice_rev * 3, 2),
            voice_loss_4h=round(voice_rev * 4, 2),
            total_loss_1h=round(total_rev * 1, 2),
            total_loss_2h=round(total_rev * 2, 2),
            total_loss_3h=round(total_rev * 3, 2),
            total_loss_4h=round(total_rev * 4, 2),
            priority_score=round(total_rev * 4 + state_bonus, 2),
        ))
        count += 1
    db.commit()
    return count

def prediction_rows(db: Session) -> list[dict]:
    rows = db.query(Prediction).order_by(Prediction.priority_score.desc()).all()
    return [{
        "site_name": r.site_name,
        "region": r.region,
        "alarm_state": r.alarm_state,
        "event_time": r.event_time.strftime("%Y-%m-%d %H:%M") if r.event_time else "-",
        "mains_to_dc": r.predicted_mains_to_dc_min,
        "dc_to_down": r.predicted_dc_to_down_min,
        "site_down": r.predicted_site_down_duration_min,
        "data_loss_1h": r.data_loss_1h,
        "data_loss_2h": r.data_loss_2h,
        "data_loss_3h": r.data_loss_3h,
        "data_loss_4h": r.data_loss_4h,
        "voice_loss_1h": r.voice_loss_1h,
        "voice_loss_2h": r.voice_loss_2h,
        "voice_loss_3h": r.voice_loss_3h,
        "voice_loss_4h": r.voice_loss_4h,
        "total_loss_4h": r.total_loss_4h,
        "priority_score": r.priority_score,
    } for r in rows]
