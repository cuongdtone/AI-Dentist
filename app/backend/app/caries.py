"""Phát hiện vùng sâu răng: cắt từng răng (theo polygon của bước phân đoạn) -> YOLO-seg 1 lớp 'caries'
(research/caries-seg) -> polygon vùng sâu theo toạ độ ảnh gốc, gán cho răng của crop.
Cách cắt (xoay thẳng răng, lề, che ngoài mask) đọc từ <model>.json do export_onnx.py chép từ dataset,
không có file -> DEFAULT_CROP (bbox thẳng, lề 0.15 — model caries_384 đầu tiên).
"""
import json, os
import cv2
import numpy as np
from .crop_utils import DEFAULT_CROP, crop_tooth, to_image
from .model import ToothSegmenter


def load_crop_cfg(model_path):
    p = os.path.splitext(model_path)[0] + ".json"
    if os.path.exists(p):
        return {**DEFAULT_CROP, **json.load(open(p))}
    return dict(DEFAULT_CROP)


def analyze_caries(model: ToothSegmenter, crop_cfg, image, teeth, p):
    """p: conf, mask_thr, own_thr, min_area, eps.
    -> list {id, tooth_id, fdi, conf, polygon, area, tooth_frac} (sắp theo FDI)."""
    out = []
    for t in teeth:
        r = crop_tooth(image, t["polygon"], crop_cfg)
        if r is None:
            continue
        crop, M, tm = r
        found = model(crop, conf=p["conf"], mask_thr=p["mask_thr"], eps=p["eps"])
        tarea = max(int(tm.sum()), 1)
        for f in found:
            lm = np.zeros(tm.shape, np.uint8)
            cv2.fillPoly(lm, [np.round(np.asarray(f["polygon"])).astype(np.int32)], 1)
            area = int(lm.sum())
            # chỉ giữ vùng sâu nằm chủ yếu trong răng của crop (răng kề tự xử lý ở crop của nó)
            if area == 0 or (lm.astype(bool) & tm).sum() / area < p["own_thr"]:
                continue
            area_img = area / abs(np.linalg.det(M[:, :2]))  # crop cùng tỉ lệ ảnh gốc (chỉ xoay) -> det = 1
            if area_img < p["min_area"]:
                continue
            out.append({
                "tooth_id": t.get("id"), "fdi": t["fdi"], "conf": f["conf"],
                "polygon": [[round(float(u), 1), round(float(v), 1)] for u, v in to_image(f["polygon"], M)],
                "area": int(area_img), "tooth_frac": round(area / tarea, 3),
            })
    out.sort(key=lambda c: (c["fdi"], -c["conf"]))
    for i, c in enumerate(out):
        c["id"] = i
    return out
