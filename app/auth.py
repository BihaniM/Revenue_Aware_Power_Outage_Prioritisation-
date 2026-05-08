from fastapi import Request, HTTPException
from sqlalchemy.orm import Session
from app.models import User

SESSION_USER_ID = "user_id"

def login_user(request: Request, user: User):
    request.session[SESSION_USER_ID] = user.id
    request.session["username"] = user.username
    request.session["role"] = user.role

def logout_user(request: Request):
    request.session.clear()

def current_user(request: Request, db: Session):
    user_id = request.session.get(SESSION_USER_ID)
    if not user_id:
        return None
    return db.query(User).filter(User.id == user_id).first()

def require_user(request: Request, db: Session):
    user = current_user(request, db)
    if not user:
        raise HTTPException(status_code=307, headers={"Location": "/login"})
    return user

def require_admin(request: Request, db: Session):
    user = require_user(request, db)
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user
