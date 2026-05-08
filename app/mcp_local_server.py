from __future__ import annotations
from app.database import SessionLocal, init_db
from app.models import RevenuePriority, Prediction
from app.services.analytics import site_display_name

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:
    raise RuntimeError("Install the optional MCP package with: pip install mcp") from exc

mcp = FastMCP("TelcoRevenuePriorityMCP")

@mcp.tool()
def get_top_revenue_sites(limit: int = 20) -> list[dict]:
    """Return top revenue-generating base stations from the dynamic priority table."""
    db = SessionLocal()
    try:
        rows = db.query(RevenuePriority).order_by(RevenuePriority.priority_rank.asc()).limit(limit).all()
        return [{
            "rank": r.priority_rank,
            "site_name": r.site_name,
            "display_name": site_display_name(r.site_name),
            "region": r.region,
            "hourly_total_revenue_rs": round(r.total_revenue_hour, 2),
            "alarm_state": r.alarm_state,
        } for r in rows]
    finally:
        db.close()

@mcp.tool()
def get_latest_predictions(limit: int = 20) -> list[dict]:
    """Return latest site-down prediction and revenue-loss results."""
    db = SessionLocal()
    try:
        rows = db.query(Prediction).order_by(Prediction.priority_score.desc()).limit(limit).all()
        return [{
            "site_name": r.site_name,
            "region": r.region,
            "alarm_state": r.alarm_state,
            "predicted_site_down_duration_min": r.predicted_site_down_duration_min,
            "total_loss_4h_rs": r.total_loss_4h,
            "priority_score": r.priority_score,
        } for r in rows]
    finally:
        db.close()

@mcp.tool()
def allocate_limited_generators(site_names: list[str], generator_count: int = 2) -> list[dict]:
    """Rank a given list of sites and select the first N sites for generator allocation."""
    db = SessionLocal()
    try:
        rows = db.query(RevenuePriority).filter(RevenuePriority.site_name.in_(site_names)).all()
        ranked = sorted(rows, key=lambda r: r.total_revenue_hour or 0, reverse=True)
        return [{
            "order": idx,
            "site_name": r.site_name,
            "display_name": site_display_name(r.site_name),
            "decision": "Allocate now" if idx <= generator_count else "Queue",
            "hourly_total_revenue_rs": round(r.total_revenue_hour, 2),
            "alarm_state": r.alarm_state,
        } for idx, r in enumerate(ranked, start=1)]
    finally:
        db.close()

if __name__ == "__main__":
    init_db()
    mcp.run()
