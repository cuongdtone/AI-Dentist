import json, os
import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel
from . import auth, db, pipeline

app = FastAPI(title="Tooth Diagnosis API")
auth.ensure_admin()


class LoginIn(BaseModel):
    username: str
    password: str


class DocIn(BaseModel):
    teeth: list[dict]
    strokes: list[dict] = []  # nét vẽ chú thích (bút chì / tẩy)


@app.get("/api/health")
def health():
    return {"ok": True, "model": os.path.exists(pipeline.MODEL_PATH), "steps": pipeline.STEPS}


@app.post("/api/auth/login")
def login(body: LoginIn):
    token = auth.login(body.username, body.password)
    if not token:
        raise HTTPException(401, "Sai tên đăng nhập hoặc mật khẩu")
    return {"token": token, "username": body.username}


@app.get("/api/stats")
def stats(period: str = Query("day", pattern="^(day|week|month|year)$"), user=Depends(auth.current_user)):
    return db.stats(period)


@app.post("/api/jobs")
async def create_job(file: UploadFile = File(...), user=Depends(auth.current_user)):
    data = await file.read()
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "File không phải ảnh hợp lệ")
    ext = os.path.splitext(file.filename or "")[1].lower() or ".png"
    path = f"{db.DATA_DIR}/images/{os.urandom(8).hex()}{ext}"
    with open(path, "wb") as f:
        f.write(data)
    job = pipeline.start(image, file.filename, path, user)
    return job.to_dict()


def _job(job_id):
    job = pipeline.JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Không tìm thấy job")
    return job


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str, user=Depends(auth.current_user)):
    return _job(job_id).to_dict()


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str, user=Depends(auth.current_user)):
    job = _job(job_id)
    job.cancelled = True
    return job.to_dict()


@app.put("/api/inferences/{inf_id}")
def save_edits(inf_id: int, body: DocIn, user=Depends(auth.current_user)):
    """Lưu polygon + nét vẽ đã chỉnh sửa trên UI."""
    row = db.q("SELECT result FROM inferences WHERE id=?", (inf_id,), one=True)
    if not row:
        raise HTTPException(404, "Không tìm thấy kết quả")
    res = json.loads(row["result"])
    res["teeth"], res["strokes"] = body.teeth, body.strokes
    db.q("UPDATE inferences SET result=?, teeth=? WHERE id=?", (json.dumps(res), len(body.teeth), inf_id))
    return {"ok": True}
