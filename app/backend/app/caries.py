"""Phát hiện vùng sâu răng: cắt từng răng (theo polygon của bước phân đoạn) -> YOLO-seg 1 lớp 'caries'
(research/caries-seg) -> polygon vùng sâu theo toạ độ ảnh gốc, gán cho răng chứa nó.
Cách cắt PHẢI giống research/caries-seg/prepare_data.py (bbox polygon răng + margin, không che ngoài mask).
"""
import cv2
import numpy as np
from .model import ToothSegmenter


def _fill(poly, shape):
    m = np.zeros(shape, np.uint8)
    cv2.fillPoly(m, [np.round(np.asarray(poly)).astype(np.int32)], 1)
    return m.astype(bool)


def analyze_caries(model: ToothSegmenter, image, teeth, p):
    """p: conf, mask_thr, own_thr, margin, min_area, eps.
    -> list {id, tooth_id, fdi, conf, polygon, area, tooth_frac} (sắp theo FDI)."""
    H, W = image.shape[:2]
    out = []
    for t in teeth:
        poly = np.asarray(t["polygon"], float)
        if len(poly) < 3:
            continue
        x, y = poly[:, 0], poly[:, 1]
        bw, bh = x.max() - x.min(), y.max() - y.min()
        x1 = int(max(0, np.floor(x.min() - bw * p["margin"])))
        y1 = int(max(0, np.floor(y.min() - bh * p["margin"])))
        x2 = int(min(W, np.ceil(x.max() + bw * p["margin"])))
        y2 = int(min(H, np.ceil(y.max() + bh * p["margin"])))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        crop = image[y1:y2, x1:x2]
        found = model(crop, conf=p["conf"], mask_thr=p["mask_thr"], eps=p["eps"])
        if not found:
            continue
        tm = _fill(poly - [x1, y1], crop.shape[:2])
        for f in found:
            lm = _fill(f["polygon"], crop.shape[:2])
            area = int(lm.sum())
            # crop có cả răng kề: chỉ giữ vùng sâu nằm chủ yếu trong răng này (răng kề tự bắt ở crop của nó)
            if area < p["min_area"] or (lm & tm).sum() / area < p["own_thr"]:
                continue
            out.append({
                "tooth_id": t.get("id"), "fdi": t["fdi"], "conf": f["conf"],
                "polygon": [[round(u + x1, 1), round(v + y1, 1)] for u, v in f["polygon"]],
                "area": area, "tooth_frac": round(area / max(int(tm.sum()), 1), 3),
            })
    out.sort(key=lambda c: (c["fdi"], -c["conf"]))
    for i, c in enumerate(out):
        c["id"] = i
    return out
