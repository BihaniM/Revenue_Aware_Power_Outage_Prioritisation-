from __future__ import annotations
from pathlib import Path
import io
import csv
from fastapi import FastAPI, Depends, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.middleware.sessions import SessionMiddleware
from sqlalchemy.orm import Session
from app.config import APP_NAME, APP_VERSION, APP_PATCH, APP_BUILD, APP_DEPLOYMENT_TARGET, APP_SECRET_KEY, UPLOAD_DIR
from app.database import get_db, init_db, check_database_connection
from app.models import User, RevenuePriority, Prediction, ImportLog, SiteAlarmAverage
from app.auth import login_user, logout_user, require_user, require_admin
from app.services.importer import import_kpi_data, import_power_alarms
from app.services.analytics import get_dashboard_rows, recalculate_revenue_priority
from app.services.ml_model import prediction_rows
from app.services.chat_service import answer_prompt

app = FastAPI(title=APP_NAME)
app.add_middleware(SessionMiddleware, secret_key=APP_SECRET_KEY)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")

def template_defaults(request: Request) -> dict:
    return {
        "app_name": APP_NAME,
        "app_version": APP_VERSION,
        "app_patch": APP_PATCH,
        "app_build": APP_BUILD,
        "app_deployment_target": APP_DEPLOYMENT_TARGET,
    }

templates = Jinja2Templates(
    directory=Path(__file__).parent / "templates",
    context_processors=[template_defaults],
)

@app.on_event("startup")
def startup_event():
    init_db()

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "error": None})

@app.post("/login")
def login_submit(request: Request, username: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == username.strip()).first()
    if not user or user.password_plain != password:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid username or password"}, status_code=401)
    login_user(request, user)
    return RedirectResponse("/", status_code=303)

@app.get("/logout")
def logout(request: Request):
    logout_user(request)
    return RedirectResponse("/login", status_code=303)

@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    user = require_user(request, db)
    top_rows = get_dashboard_rows(db, 20)
    preds = prediction_rows(db)
    averages_count = db.query(SiteAlarmAverage).count()
    kpi_count = db.query(RevenuePriority).count()
    last_imports = db.query(ImportLog).order_by(ImportLog.created_at.desc()).limit(5).all()
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "top_rows": top_rows,
        "predictions": preds,
        "averages_count": averages_count,
        "kpi_count": kpi_count,
        "last_imports": last_imports,
    })

@app.post("/upload/kpi")
async def upload_kpi(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    require_user(request, db)
    if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Upload a CSV or Excel file")
    saved = UPLOAD_DIR / file.filename
    saved.write_bytes(await file.read())
    count = import_kpi_data(db, saved, import_tag="uploaded", recalculate=True)
    return RedirectResponse(f"/?message=Imported {count} KPI records", status_code=303)

@app.post("/upload/alarm")
async def upload_alarm(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    require_user(request, db)
    if not file.filename.lower().endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Upload a CSV or Excel file")
    saved = UPLOAD_DIR / file.filename
    saved.write_bytes(await file.read())
    count = import_power_alarms(db, saved, import_tag="uploaded", is_training=False)
    recalculate_revenue_priority(db)
    return RedirectResponse(f"/?message=Imported {count} power alarm records", status_code=303)

@app.get("/admin", response_class=HTMLResponse)
def admin_page(request: Request, db: Session = Depends(get_db)):
    user = require_admin(request, db)
    users = db.query(User).order_by(User.id.asc()).all()
    imports = db.query(ImportLog).order_by(ImportLog.created_at.desc()).limit(30).all()
    return templates.TemplateResponse("admin.html", {"request": request, "user": user, "users": users, "imports": imports})

@app.post("/admin/users/create")
def create_user(request: Request, username: str = Form(...), role: str = Form("common"), db: Session = Depends(get_db)):
    require_admin(request, db)
    if db.query(User).filter(User.username == username.strip()).first():
        return RedirectResponse("/admin?error=Username already exists", status_code=303)
    db.add(User(username=username.strip(), password_plain="Changeme_123", role=role))
    db.commit()
    return RedirectResponse("/admin?message=User created with default password Changeme_123", status_code=303)

@app.post("/admin/users/update")
def update_user(request: Request, user_id: int = Form(...), password: str = Form(...), role: str = Form("common"), db: Session = Depends(get_db)):
    require_admin(request, db)
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.password_plain = password
    user.role = role
    db.commit()
    return RedirectResponse("/admin?message=User updated", status_code=303)

@app.post("/admin/users/delete")
def delete_user(request: Request, user_id: int = Form(...), db: Session = Depends(get_db)):
    current = require_admin(request, db)
    if current.id == user_id:
        return RedirectResponse("/admin?error=Admin cannot delete own account while logged in", status_code=303)
    user = db.query(User).filter(User.id == user_id).first()
    if user:
        db.delete(user)
        db.commit()
    return RedirectResponse("/admin?message=User deleted", status_code=303)

@app.post("/api/chat")
def chat(request: Request, prompt: str = Form(...), db: Session = Depends(get_db)):
    require_user(request, db)
    answer = answer_prompt(prompt, db)
    return {"answer": answer}

@app.get("/export/predictions.csv")
def export_predictions(request: Request, db: Session = Depends(get_db)):
    require_user(request, db)
    output = io.StringIO()
    writer = csv.writer(output)
    header = [
        "Site Name", "Region", "Alarm State", "Event Time", "Mains to DC (min)", "DC to Down (min)",
        "Predicted Site Down Duration (min)", "Data Loss 1h", "Data Loss 2h", "Data Loss 3h", "Data Loss 4h",
        "Voice Loss 1h", "Voice Loss 2h", "Voice Loss 3h", "Voice Loss 4h", "Total Loss 4h", "Priority Score"
    ]
    writer.writerow(header)
    for r in db.query(Prediction).order_by(Prediction.priority_score.desc()).all():
        writer.writerow([
            r.site_name, r.region, r.alarm_state, r.event_time, r.predicted_mains_to_dc_min, r.predicted_dc_to_down_min,
            r.predicted_site_down_duration_min, r.data_loss_1h, r.data_loss_2h, r.data_loss_3h, r.data_loss_4h,
            r.voice_loss_1h, r.voice_loss_2h, r.voice_loss_3h, r.voice_loss_4h, r.total_loss_4h, r.priority_score
        ])
    output.seek(0)
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=prediction_results.csv"})

@app.get("/health")
def health():
    db_status = check_database_connection()
    return {
        "status": "ok",
        "app": APP_NAME,
        "version": APP_VERSION,
        "patch": APP_PATCH,
        "build": APP_BUILD,
        "deployment_target": APP_DEPLOYMENT_TARGET,
        "database": db_status,
    }
