from __future__ import annotations

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from app.config import DATABASE_URL

# SQLite is retained for local/offline testing. For deployment, use Supabase PostgreSQL
# by setting DATABASE_URL in .env, for example:
# postgresql+psycopg2://postgres.PROJECT_REF:DB_PASSWORD@aws-0-REGION.pooler.supabase.com:5432/postgres?sslmode=require

engine_kwargs = {"pool_pre_ping": True}

if DATABASE_URL.startswith("sqlite"):
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    # Helpful for managed PostgreSQL/Supabase connections behind a pooler.
    engine_kwargs.update({
        "pool_size": 5,
        "max_overflow": 10,
        "pool_recycle": 1800,
    })

engine = create_engine(DATABASE_URL, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    from app import models
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        admin = db.query(models.User).filter(models.User.username == "admin").first()
        if not admin:
            db.add(models.User(username="admin", password_plain="admin", role="admin"))
            db.commit()
    finally:
        db.close()

def check_database_connection() -> dict:
    """Small helper used by deployment checks and health endpoint."""
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    db_type = "sqlite" if DATABASE_URL.startswith("sqlite") else "postgresql/supabase-compatible"
    safe_url = DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else DATABASE_URL
    return {"status": "ok", "database_type": db_type, "database_target": safe_url}
