"""Pipeline chẩn đoán: chuỗi bước tuần tự, chạy trong thread, hỗ trợ huỷ giữa các bước.
Thêm bước mới: viết hàm f(image, result) rồi đăng ký vào RUNNERS.
"""
import os, threading, time, uuid
from collections import OrderedDict
import cv2
import numpy as np
from . import caries, db, wisdom_geom
from .model import ToothSegmenter

MODEL_PATH = os.getenv("MODEL_PATH", "./models/tooth_seg.onnx")
CARIES_MODEL_PATH = os.getenv("CARIES_MODEL_PATH", "./models/caries_seg.onnx")
# Ngưỡng luật ngầm/lệch, chỉnh trên DENTEX bằng research/wisdom-analysis/calibrate.py
WISDOM_PARAMS = wisdom_geom.load_params(os.path.join(os.path.dirname(__file__), "wisdom_params.json"))

STEPS = [
    ("seg", "Phân đoạn răng"),
    ("wisdom", "Phân tích răng khôn"),
    ("caries", "Phân tích sâu răng"),
    ("perio", "Phân tích đường nha chu"),
    ("cej", "Phân tích mức CEJ"),
]

_models = {}
_model_lock = threading.Lock()


class Skip(Exception):
    """Bước không chạy được nhưng không phải lỗi (vd chưa có model) -> status 'skipped' + note."""


def _load(path) -> ToothSegmenter:
    with _model_lock:
        if path not in _models:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Không thấy model: {path}")
            _models[path] = ToothSegmenter(path)
        return _models[path]


def get_model() -> ToothSegmenter:
    return _load(MODEL_PATH)


def get_caries_model():
    """-> (model, cấu hình cắt răng lúc train)."""
    if not os.path.exists(CARIES_MODEL_PATH):
        raise Skip("Chưa có model sâu răng")
    return _load(CARIES_MODEL_PATH), caries.load_crop_cfg(CARIES_MODEL_PATH)


# ---------------- Cấu hình từng bước (UI tự sinh form từ schema này) ----------------
_n = lambda key, label, lo, hi, step, default, hint: dict(key=key, label=label, type="number", min=lo, max=hi, step=step, default=default, hint=hint)
CONFIG_SCHEMA = {
    "seg": [
        _n("conf", "Ngưỡng tin cậy (conf)", 0.05, 0.95, 0.05, 0.25,
           "Chỉ giữ răng có độ tin cậy ≥ ngưỡng. Tăng → ít phát hiện sai nhưng dễ bỏ sót răng mờ, gãy, mất một phần; "
           "giảm → bắt được nhiều răng hơn nhưng có thể sinh răng ảo."),
        _n("iou", "Ngưỡng IoU (NMS)", 0.1, 0.9, 0.05, 0.5,
           "Hai khung chồng nhau với IoU (giao/hợp) lớn hơn ngưỡng bị coi là cùng một răng, chỉ giữ khung conf cao hơn. "
           "Giảm → loại trùng mạnh (có thể mất răng mọc chen chúc); tăng → giữ nhiều khung chồng nhau hơn."),
        _n("mask_thr", "Ngưỡng mask", 0.1, 0.9, 0.05, 0.5,
           "Xác suất tối thiểu để một điểm ảnh được tính là thuộc răng. Tăng → viền răng co vào, gọn hơn; giảm → viền nở ra."),
        _n("ios_thr", "Ngưỡng loại mask trùng", 0.3, 1.0, 0.05, 0.6,
           "Tỉ lệ (diện tích giao / diện tích mask nhỏ hơn) giữa hai mask. Nếu ≥ ngưỡng, mask conf thấp hơn bị bỏ — "
           "bắt trường hợp mask nhỏ nằm gọn trong mask to mà NMS bỏ sót. Đặt 1.0 để tắt."),
        _n("eps", "Độ rút gọn polygon (px)", 0.5, 5.0, 0.5, 1.0,
           "Sai số tối đa (pixel, trên ảnh 1024) khi rút gọn viền mask thành polygon. "
           "Lớn → ít đỉnh, dễ chỉnh sửa nhưng kém sát viền; nhỏ → nhiều đỉnh, sát viền hơn."),
    ],
    "wisdom": [
        dict(key="jaws", label="Hàm phân tích", type="select", default="both",
             options=[{"value": "both", "label": "Cả hai hàm (18, 28, 38, 48)"},
                      {"value": "lower", "label": "Chỉ hàm dưới (38, 48)"},
                      {"value": "upper", "label": "Chỉ hàm trên (18, 28)"}],
             hint="Chọn răng khôn cần phân tích. Phân loại Pell & Gregory chủ yếu dùng cho hàm dưới; hàm trên dùng tương tự theo Archer."),
        _n("angle_thr", "Ngưỡng góc ngầm/lệch (°)", 0, 90, 1, WISDOM_PARAMS["angle_thr"],
           "Góc Winter = góc giữa trục răng khôn và trục răng 7 cùng bên (dương = nghiêng về phía gần). "
           "|góc| > ngưỡng → kết luận NGẦM/LỆCH. Mặc định đã hiệu chỉnh trên dữ liệu DENTEX."),
        _n("impact_depth", "Ngưỡng độ sâu ngầm", 0, 1, 0.01, WISDOM_PARAMS["impact_depth"],
           "Độ sâu r = (mặt nhai răng 7 − điểm cao nhất răng khôn) / chiều dài răng 7. r > ngưỡng → NGẦM/LỆCH. "
           "Ví dụ 0.19: răng khôn thấp hơn mặt nhai 19% chiều dài răng 7. Mặc định đã hiệu chỉnh trên DENTEX."),
        _n("depth_a", "Dung sai độ sâu loại A", 0, 0.3, 0.01, WISDOM_PARAMS["depth_a"],
           "Loại A (Pell & Gregory): điểm cao nhất răng khôn ngang hoặc cao hơn mặt nhai răng 7. "
           "r ≤ giá trị này vẫn được xếp loại A (bù sai số mask)."),
        _n("cej_frac", "Vị trí CEJ răng 7", 0.1, 0.6, 0.01, WISDOM_PARAMS["cej_frac"],
           "Mask không có mốc CEJ (đường nối men – xê-măng) nên CEJ được ước lượng cách mặt nhai = giá trị × chiều dài răng 7. "
           "Là ranh giới loại B / C: r ≤ giá trị → B, lớn hơn → C."),
        _n("min_elong", "Độ thon dài tối thiểu", 1.0, 2.0, 0.05, WISDOM_PARAMS["min_elong"],
           "Tỉ lệ trục dài / trục ngắn của mask. Dưới ngưỡng (răng gần tròn, ví dụ mầm răng chưa có chân) "
           "thì trục răng không đáng tin → kết quả bị gắn cờ cảnh báo."),
    ],
    "caries": [
        _n("conf", "Ngưỡng tin cậy (conf)", 0.05, 0.95, 0.05, 0.25,
           "Chỉ giữ vùng sâu có độ tin cậy ≥ ngưỡng. Tăng → ít báo sai (vết mờ, chồng ảnh, bóng miếng trám) nhưng dễ "
           "bỏ sót sâu nhỏ/sớm; giảm → bắt nhiều hơn nhưng nhiều báo động giả."),
        _n("mask_thr", "Ngưỡng mask", 0.1, 0.9, 0.05, 0.5,
           "Xác suất tối thiểu để một điểm ảnh thuộc vùng sâu. Tăng → viền vùng sâu co lại; giảm → nở ra."),
        _n("own_thr", "Tỉ lệ nằm trong răng", 0.1, 1.0, 0.05, 0.5,
           "Mỗi răng được xoay thẳng và cắt riêng, nhưng mép khung vẫn lẫn một phần răng kề. Vùng sâu chỉ được gán "
           "cho răng nếu ≥ tỉ lệ này diện tích nằm trong viền răng đó. Giảm nếu vùng sâu ở mặt tiếp giáp bị mất "
           "(viền răng của model hơi hụt); tăng nếu một vùng sâu bị báo cho cả hai răng. "
           "Cách cắt (xoay, lề) cố định theo lúc train, không chỉnh ở đây."),
        _n("min_area", "Diện tích tối thiểu (px)", 0, 2000, 10, 50,
           "Bỏ vùng sâu nhỏ hơn số pixel này (trên ảnh gốc) — thường là nhiễu."),
        _n("eps", "Độ rút gọn polygon (px)", 0.5, 5.0, 0.5, 1.0,
           "Sai số tối đa (pixel, trên ảnh crop đã đưa về kích thước model) khi rút gọn viền vùng sâu thành polygon."),
    ],
}
REQUIRES = {"wisdom": "seg", "caries": "seg", "perio": "seg", "cej": "seg"}  # bước cần kết quả của bước khác
WISDOM_JAWS = {"both": (18, 28, 38, 48), "lower": (38, 48), "upper": (18, 28)}


def resolve_config(raw: dict | None) -> dict:
    """Ghép config từ client với mặc định, ép kiểu + kẹp trong [min, max]. -> {step: {enabled, <param>...}}"""
    raw = raw or {}
    out = {}
    for key, _ in STEPS:
        r = raw.get(key) or {}
        c = {"enabled": bool(r.get("enabled", True))}
        for f in CONFIG_SCHEMA.get(key, []):
            v = r.get(f["key"], f["default"])
            if f["type"] == "number":
                try:
                    v = min(f["max"], max(f["min"], float(v)))
                except (TypeError, ValueError):
                    v = f["default"]
            elif f["type"] == "select" and v not in [o["value"] for o in f["options"]]:
                v = f["default"]
            c[f["key"]] = v
        out[key] = c
    return out


def step_seg(image, res, p):
    res["teeth"] = get_model()(image, conf=p["conf"], iou=p["iou"], mask_thr=p["mask_thr"], ios_thr=p["ios_thr"], eps=p["eps"])


def analyze_wisdom(teeth, width, height, p=None):
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
    params = {**WISDOM_PARAMS, **{k: v for k, v in (p or {}).items() if k in WISDOM_PARAMS}}
    for r in wisdom_geom.analyze_teeth(masks, params, WISDOM_JAWS[(p or {}).get("jaws", "both")]):
        r["fdi"] = str(r["fdi"])
        r["tooth_id"] = ids.get(int(r["fdi"]))
        r["text"] = wisdom_geom.describe({**r, "fdi": int(r["fdi"])})
        out.append(r)
    return out


def step_wisdom(image, res, p):
    res["findings"]["wisdom"] = analyze_wisdom(res["teeth"], res["width"], res["height"], p)


def step_caries(image, res, p):
    model, crop_cfg = get_caries_model()
    res["findings"]["caries"] = caries.analyze_caries(model, crop_cfg, image, res["teeth"], p)


RUNNERS = {"seg": step_seg, "wisdom": step_wisdom, "caries": step_caries}  # bước chưa có -> "skipped"


def counts(res):
    f = res["findings"]
    return {"teeth": len(res["teeth"]), **{k: len(f.get(k, [])) for k in ("wisdom", "caries", "perio", "cej")}}


class Job:
    def __init__(self, image, filename, image_path, username, config=None):
        self.id = uuid.uuid4().hex
        self.image, self.filename, self.image_path, self.username = image, filename, image_path, username
        self.status = "running"  # running | done | error | cancelled
        self.cancelled = False
        self.inference_id = None
        self.config = resolve_config(config)
        self.steps = [{"key": k, "name": n, "status": "pending", "ms": None, "error": None, "note": None} for k, n in STEPS]
        h, w = image.shape[:2]
        self.result = {"width": w, "height": h, "teeth": [], "findings": {}, "config": self.config}

    def run(self):
        for st in self.steps:
            if self.cancelled:
                self.status = "cancelled"
                return
            cfg = self.config[st["key"]]
            if not cfg["enabled"]:
                st["status"] = "disabled"
                continue
            fn = RUNNERS.get(st["key"])
            if fn is None:
                st["status"] = "skipped"
                continue
            req = next((s for s in self.steps if s["key"] == REQUIRES.get(st["key"])), None)
            if req and req["status"] != "done":
                st.update(status="skipped", note=f"Cần bước {req['name']}")
                continue
            st["status"] = "running"
            t0 = time.perf_counter()
            try:
                fn(self.image, self.result, cfg)
            except Skip as e:
                st.update(status="skipped", note=str(e))
                continue
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


def start(image, filename, image_path, username, config=None) -> Job:
    job = Job(image, filename, image_path, username, config)
    JOBS[job.id] = job
    while len(JOBS) > MAX_JOBS:
        JOBS.popitem(last=False)
    threading.Thread(target=job.run, daemon=True).start()
    return job
