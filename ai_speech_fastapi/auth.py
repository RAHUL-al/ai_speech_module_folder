import os, datetime
from jose import jwt, JWTError
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi_sqlalchemy import db
from models import User
from schemas import UserOut

SECRET_KEY     = os.getenv("JWT_SECRET", "unsafe")
ALGORITHM      = os.getenv("JWT_ALGORITHM", "HS256")
EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", 60))

pwd_ctx  = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2   = OAuth2PasswordBearer(tokenUrl="/login")
_blacklist: set[str] = set()

# ---------------- password helpers ----------------
def hash_password(password: str) -> str:
    return pwd_ctx.hash(password)

def verify_password(password: str, hashed: str) -> bool:
    return pwd_ctx.verify(password, hashed)

# ---------------- JWT helpers ---------------------
def create_access_token(data: dict) -> str:
    payload = data.copy()
    expire  = datetime.datetime.utcnow() + datetime.timedelta(minutes=EXPIRE_MINUTES)
    payload.update({"exp": expire})
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid token")

# -------------- dependency: current user ----------
def get_current_user(token: str = Depends(oauth2)) -> UserOut:
    if token in _blacklist:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Token revoked. Log in again.")
    payload = decode_token(token)
    user_id = payload.get("sub")
    user    = db.session.get(User, int(user_id))
    if not user:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "User not found")
    return UserOut.from_orm(user)

# -------------- logout helper ---------------------
def blacklist_token(token: str):
    _blacklist.add(token)



