import hashlib, hmac, os, secrets
from datetime import datetime, timedelta, timezone
import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from . import db

SECRET = os.getenv("JWT_SECRET", "change-me")
TTL_HOURS = 12


def _hash(pw: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), 200_000).hex()


def create_user(username: str, password: str):
    salt = secrets.token_hex(16)
    db.q("INSERT OR REPLACE INTO users VALUES (?,?,?)", (username, salt, _hash(password, salt)))


def ensure_admin():
    user, pw = os.getenv("ADMIN_USER", "admin"), os.getenv("ADMIN_PASSWORD", "admin")
    if not db.q("SELECT 1 FROM users WHERE username=?", (user,), one=True):
        create_user(user, pw)


def login(username: str, password: str) -> str | None:
    u = db.q("SELECT * FROM users WHERE username=?", (username,), one=True)
    if not u or not hmac.compare_digest(_hash(password, u["salt"]), u["pw_hash"]):
        return None
    exp = datetime.now(timezone.utc) + timedelta(hours=TTL_HOURS)
    return jwt.encode({"sub": username, "exp": exp}, SECRET, algorithm="HS256")


def current_user(cred: HTTPAuthorizationCredentials = Depends(HTTPBearer())) -> str:
    try:
        return jwt.decode(cred.credentials, SECRET, algorithms=["HS256"])["sub"]
    except jwt.PyJWTError:
        raise HTTPException(401, "Phiên đăng nhập không hợp lệ")
