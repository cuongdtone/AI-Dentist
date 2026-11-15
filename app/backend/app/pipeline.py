"""Pipeline chẩn đoán: chuỗi bước tuần tự, chạy trong thread, hỗ trợ huỷ giữa các bước.
Thêm bước mới: viết hàm f(image, result) rồi đăng ký vào RUNNERS.
"""
import os, threading, time, uuid
from collections import OrderedDict
from . import db
from .model import ToothSegmenter

MODEL_PATH = os.getenv("MODEL_PATH", "./models/tooth_seg.onnx")
WISDOM_FDI = {"18", "28", "38", "48"}

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


def step_wisdom(image, res):
    res["findings"]["wisdom"] = [t["fdi"] for t in res["teeth"] if t["fdi"] in WISDOM_FDI]


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
