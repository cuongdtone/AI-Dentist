"""Pipeline chẩn đoán: chuỗi bước tuần tự, chạy trong thread, hỗ trợ huỷ giữa các bước.
Thêm bước mới: viết hàm f(image, result) rồi đăng ký vào RUNNERS.
"""
import os, threading, time, uuid
from collections import OrderedDict
import cv2
import numpy as np
from . import db, wisdom_geom
from .model import ToothSegmenter

MODEL_PATH = os.getenv("MODEL_PATH", "./models/tooth_seg.onnx")
# Ngưỡng luật ngầm/lệch, chỉnh trên DENTEX bằng research/wisdom-analysis/calibrate.py
WISDOM_PARAMS = wisdom_geom.load_params(os.path.join(os.path.dirname(__file__), "wisdom_params.json"))

STEPS = [
    ("seg", "Phân đoạn răng"),
    ("wisdom", "Phân tích răng khôn"),
    ("caries", "Phân tích sâu răng"),
    ("perio", "Phân tích đường nha chu"),
    ("cej", "Phân tích mức CEJ"),
]

_model = None
_model_lock = threading.Lock()


def get_model() -> ToothSegmenter:
    global _model
    with _model_lock:
        if _model is None:
            if not os.path.exists(MODEL_PATH):
                raise FileNotFoundError(f"Không thấy model: {MODEL_PATH}")
            _model = ToothSegmenter(MODEL_PATH)
        return _model


def step_seg(image, res):
    res["teeth"] = get_model()(image)


def analyze_wisdom(teeth, width, height):
    """Phân tích hình học răng khôn từ polygon răng 7 & 8: Winter, độ sâu A/B/C, ngầm/lệch.
    Trả về list (mỗi phần tử 1 răng khôn), thêm "text" mô tả tiếng Việt và "tooth_id" để UI hover."""
    masks, ids, best = {}, {}, {}
    for t in teeth:
        f = int(t["fdi"])
        # trùng số FDI (model đánh nhầm số) -> chỉ phân tích răng conf cao nhất
        if f % 10 in (7, 8) and len(t["polygon"]) >= 3 and t.get("conf", 1) > best.get(f, -1):
            best[f] = t.get("conf", 1)
            m = np.zeros((height, width), np.uint8)
            cv2.fillPoly(m, [np.round(np.array(t["polygon"])).astype(np.int32)], 1)
            masks[f], ids[f] = m.astype(bool), t.get("id")
    out = []
    for r in wisdom_geom.analyze_teeth(masks, WISDOM_PARAMS):
        r["fdi"] = str(r["fdi"])
        r["tooth_id"] = ids.get(int(r["fdi"]))
        r["text"] = wisdom_geom.describe({**r, "fdi": int(r["fdi"])})
        out.append(r)
    return out


def step_wisdom(image, res):
    res["findings"]["wisdom"] = analyze_wisdom(res["teeth"], res["width"], res["height"])


RUNNERS = {"seg": step_seg, "wisdom": step_wisdom}  # bước chưa có -> "skipped"


def counts(res):
    f = res["findings"]
    return {"teeth": len(res["teeth"]), **{k: len(f.get(k, [])) for k in ("wisdom", "caries", "perio", "cej")}}


class Job:
    def __init__(self, image, filename, image_path, username):
        self.id = uuid.uuid4().hex
        self.image, self.filename, self.image_path, self.username = image, filename, image_path, username
        self.status = "running"  # running | done | error | cancelled
        self.cancelled = False
        self.inference_id = None
        self.steps = [{"key": k, "name": n, "status": "pending", "ms": None, "error": None} for k, n in STEPS]
        h, w = image.shape[:2]
        self.result = {"width": w, "height": h, "teeth": [], "findings": {}}

    def run(self):
        for st in self.steps:
            if self.cancelled:
                self.status = "cancelled"
                return
            fn = RUNNERS.get(st["key"])
            if fn is None:
                st["status"] = "skipped"
                continue
            st["status"] = "running"
            t0 = time.perf_counter()
            try:
                fn(self.image, self.result)
            except Exception as e:
                st.update(status="error", error=str(e))
                self.status = "error"
                return
            st.update(status="done", ms=round((time.perf_counter() - t0) * 1000, 1))
        self.inference_id = db.save_inference(self.username, self.filename, self.image_path, self.result, counts(self.result))
        self.image = None  # giải phóng bộ nhớ
        self.status = "done"

    def to_dict(self):
        return {"id": self.id, "status": self.status, "steps": self.steps,
                "result": self.result, "inference_id": self.inference_id}


JOBS: "OrderedDict[str, Job]" = OrderedDict()
MAX_JOBS = 50


def start(image, filename, image_path, username) -> Job:
    job = Job(image, filename, image_path, username)
    JOBS[job.id] = job
    while len(JOBS) > MAX_JOBS:
        JOBS.popitem(last=False)
    threading.Thread(target=job.run, daemon=True).start()
    return job
