from __future__ import annotations
from datetime import datetime
import pandas as pd
from sqlalchemy.orm import Session
from app.config import DATA_CHARGE_PER_MB, VOICE_CHARGE_PER_MIN
from app.models import PowerAlarm, KpiData, SelectedSite, SiteAlarmAverage, RevenuePriority

ALARM_STATE_BY_SEQUENCE = {
    1: "Power failure",
    2: "DC low alarm",
    3: "Site down alarm",
}
STATE_WEIGHT = {
    "Normal": 0,
    "Power failure": 1,
    "DC low alarm": 2,
    "Site down alarm": 3,
}

def site_display_name(site_name: str) -> str:
    if "_G0L00_" in site_name:
        name = site_name.split("_G0L00_", 1)[1]
    else:
        name = site_name
    for token in ["_IP", "_TXD", "_NOA", "__IP", "_9", "_#", "_Wa", "_Ra", "_Pi", "_Ab", "_Gm", "_Di", "_Ne"]:
        name = name.replace(token, "_")
    return name.replace("_", " ").replace("#", "").strip()

def _alarms_dataframe(db: Session) -> pd.DataFrame:
    rows = db.query(PowerAlarm).all()
    return pd.DataFrame([{
        "incident_id": r.incident_id,
        "site_name": r.site_name,
        "region": r.region,
        "alarm_sequence": r.alarm_sequence,
        "alarm_name": r.alarm_name,
        "occurred_on": r.occurred_on,
        "cleared_on": r.cleared_on,
        "alarm_duration_min": r.alarm_duration_min,
    } for r in rows])

def _kpi_dataframe(db: Session) -> pd.DataFrame:
    rows = db.query(KpiData).all()
    return pd.DataFrame([{
        "start_time": r.start_time,
        "site_name": r.site_name,
        "data_traffic_mb": r.data_traffic_mb,
        "voice_traffic_erl": r.voice_traffic_erl,
        "hour_of_day": r.hour_of_day,
    } for r in rows])

def _remove_iqr_outliers(group: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    clean = group.copy()
    for col in cols:
        if len(clean) < 4:
            continue
        q1 = clean[col].quantile(0.25)
        q3 = clean[col].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        clean = clean[(clean[col] >= q1 - 1.5 * iqr) & (clean[col] <= q3 + 1.5 * iqr)]
    return clean if len(clean) else group

def recalculate_site_alarm_averages(db: Session) -> None:
    df = _alarms_dataframe(db)
    if df.empty:
        return
    rows = []
    for (incident_id, site_name), g in df.groupby(["incident_id", "site_name"]):
        by_name = {str(r.alarm_name).lower(): r for r in g.itertuples()}
        mains = by_name.get("mains input out of range")
        dc = by_name.get("power supply dc output out of range")
        down = by_name.get("enodeb out of service")
        if not mains:
            continue
        region = getattr(mains, "region", "")
        mains_to_dc = None
        dc_to_down = None
        site_down_duration = None
        if mains and dc:
            mains_to_dc = (dc.occurred_on - mains.occurred_on).total_seconds() / 60
        if dc and down:
            dc_to_down = (down.occurred_on - dc.occurred_on).total_seconds() / 60
        if down:
            site_down_duration = float(down.alarm_duration_min or 0)
            if site_down_duration == 0 and down.cleared_on:
                site_down_duration = (down.cleared_on - down.occurred_on).total_seconds() / 60
        rows.append({
            "site_name": site_name,
            "region": region,
            "mains_to_dc": mains_to_dc,
            "dc_to_down": dc_to_down,
            "site_down_duration": site_down_duration,
        })
    if not rows:
        return
    agg = pd.DataFrame(rows).groupby(["site_name", "region"], dropna=False).agg(
        avg_mains_to_dc_min=("mains_to_dc", "mean"),
        avg_dc_to_down_min=("dc_to_down", "mean"),
        avg_site_down_duration_min=("site_down_duration", "mean"),
        sample_count=("site_name", "count"),
    ).reset_index()
    db.query(SiteAlarmAverage).delete()
    for row in agg.to_dict("records"):
        db.add(SiteAlarmAverage(
            site_name=row["site_name"],
            region=row["region"],
            avg_mains_to_dc_min=float(row["avg_mains_to_dc_min"] or 0),
            avg_dc_to_down_min=float(row["avg_dc_to_down_min"] or 0),
            avg_site_down_duration_min=float(row["avg_site_down_duration_min"] or 0),
            sample_count=int(row["sample_count"]),
            updated_at=datetime.utcnow(),
        ))
    db.commit()

def recalculate_revenue_priority(db: Session, reference_time: datetime | None = None) -> None:
    df = _kpi_dataframe(db)
    if df.empty:
        return
    df["start_time"] = pd.to_datetime(df["start_time"])
    max_time = reference_time or df["start_time"].max().to_pydatetime()
    start_window = pd.Timestamp(max_time) - pd.Timedelta(days=7)
    current_hour = int(pd.Timestamp(max_time).hour)
    latest = df[(df["start_time"] >= start_window) & (df["hour_of_day"] == current_hour)]
    if latest.empty:
        latest = df[df["start_time"] >= start_window]
    sites = {s.site_name: s.region for s in db.query(SelectedSite).all()}
    current_state = {p.site_name: (p.alarm_state, p.last_alarm_time) for p in db.query(RevenuePriority).all()}
    result = []
    for site_name, g in latest.groupby("site_name"):
        clean = _remove_iqr_outliers(g, ["data_traffic_mb", "voice_traffic_erl"])
        avg_data = float(clean["data_traffic_mb"].mean())
        avg_voice = float(clean["voice_traffic_erl"].mean())
        data_rev = avg_data * DATA_CHARGE_PER_MB
        voice_rev = avg_voice * 60 * VOICE_CHARGE_PER_MIN
        state, last_alarm = current_state.get(site_name, ("Normal", None))
        result.append({
            "site_name": site_name,
            "region": sites.get(site_name, ""),
            "hour_of_day": current_hour,
            "avg_data_mb_per_hour": avg_data,
            "avg_voice_erl_per_hour": avg_voice,
            "data_revenue_hour": data_rev,
            "voice_revenue_hour": voice_rev,
            "total_revenue_hour": data_rev + voice_rev,
            "alarm_state": state,
            "last_alarm_time": last_alarm,
        })
    ranked = sorted(result, key=lambda x: x["total_revenue_hour"], reverse=True)
    db.query(RevenuePriority).delete()
    for idx, row in enumerate(ranked, start=1):
        db.add(RevenuePriority(priority_rank=idx, updated_at=datetime.utcnow(), **row))
    db.commit()

def apply_alarm_states_from_dataframe(db: Session, df: pd.DataFrame) -> None:
    for priority in db.query(RevenuePriority).all():
        priority.alarm_state = "Normal"
        priority.last_alarm_time = None
    if df.empty:
        db.commit()
        return
    df = df.copy()
    df["occurred_on"] = pd.to_datetime(df["occurred_on"], errors="coerce")
    df["alarm_sequence"] = pd.to_numeric(df["alarm_sequence"], errors="coerce").fillna(0).astype(int)
    latest = df.sort_values(["site_name", "occurred_on", "alarm_sequence"]).groupby("site_name").tail(1)
    for row in latest.to_dict("records"):
        priority = db.query(RevenuePriority).filter(RevenuePriority.site_name == row["site_name"]).first()
        if not priority:
            priority = RevenuePriority(site_name=row["site_name"], region=row.get("region", ""), priority_rank=9999, hour_of_day=int(row["occurred_on"].hour if pd.notna(row["occurred_on"]) else 0))
            db.add(priority)
        priority.alarm_state = ALARM_STATE_BY_SEQUENCE.get(int(row.get("alarm_sequence", 1)), "Power failure")
        priority.last_alarm_time = row["occurred_on"].to_pydatetime() if hasattr(row["occurred_on"], "to_pydatetime") else row["occurred_on"]
    db.commit()

def get_dashboard_rows(db: Session, limit: int = 20) -> list[dict]:
    rows = db.query(RevenuePriority).order_by(RevenuePriority.priority_rank.asc()).limit(limit).all()
    return [{
        "rank": r.priority_rank,
        "site_name": r.site_name,
        "display_name": site_display_name(r.site_name),
        "region": r.region,
        "hour": r.hour_of_day,
        "data_revenue": round(r.data_revenue_hour, 2),
        "voice_revenue": round(r.voice_revenue_hour, 2),
        "total_revenue": round(r.total_revenue_hour, 2),
        "alarm_state": r.alarm_state,
        "state_class": r.alarm_state.lower().replace(" ", "-"),
        "last_alarm_time": r.last_alarm_time.strftime("%Y-%m-%d %H:%M") if r.last_alarm_time else "-",
    } for r in rows]
