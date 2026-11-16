"""Cắt 1 răng từ phim panoramic, DÙNG CHUNG cho tạo dataset (research) và app (app/backend/app/crop_utils.py là bản sao).

align=True: xoay để trục dài răng (PCA trên polygon đã tô) thẳng đứng rồi cắt khung sát răng (+margin)
            -> răng chính nằm giữa, chiếm gần hết khung, ít răng kề.
align=False: bbox thẳng của polygon (+margin) (cách cũ).
mask_outside=True: tô đen ngoài mask răng (mask được nới mask_dilate px để giữ vùng sâu sát viền).
Trả về crop + ma trận affine M (ảnh gốc -> crop); dùng cv2.invertAffineTransform(M) để đưa kết quả về ảnh gốc.
"""
import cv2
import numpy as np

DEFAULT_CROP = {"align": False, "margin": 0.15, "mask_outside": False, "mask_dilate": 6}


def _axis_angle(poly):
    """Góc (độ) cần xoay (cv2.getRotationMatrix2D) để trục dài polygon thẳng đứng, trong (-90, 90]."""
    x1, y1 = np.floor(poly.min(0)).astype(int)
    x2, y2 = np.ceil(poly.max(0)).astype(int) + 1
    m = np.zeros((y2 - y1, x2 - x1), np.uint8)
    cv2.fillPoly(m, [np.round(poly - [x1, y1]).astype(np.int32)], 1)
    ys, xs = np.nonzero(m)
    if len(xs) < 10:
        return 0.0
    pts = np.stack([xs, ys], 1).astype(float)
    pts -= pts.mean(0)
    w, v = np.linalg.eigh(np.cov(pts.T))
    vx, vy = v[:, np.argmax(w)]  # trục dài
    # góc của trục so với phương thẳng đứng (ảnh y hướng xuống); getRotationMatrix2D: góc dương = ngược chiều kim đồng hồ
    a = -np.degrees(np.arctan2(vx, vy))
    while a > 90:
        a -= 180
    while a <= -90:
        a += 180
    return float(a)


def crop_tooth(img, poly, cfg=DEFAULT_CROP):
    """img: ảnh gốc HxWx3, poly: Nx2 (toạ độ ảnh gốc). -> (crop, M 2x3, tooth_mask_crop bool) hoặc None."""
    cfg = {**DEFAULT_CROP, **(cfg or {})}
    poly = np.asarray(poly, float)
    if len(poly) < 3:
        return None
    c = poly.mean(0)
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), _axis_angle(poly) if cfg["align"] else 0.0, 1.0)
    rp = cv2.transform(poly[None], M)[0]
    (x1, y1), (x2, y2) = rp.min(0), rp.max(0)
    mx, my = (x2 - x1) * cfg["margin"], (y2 - y1) * cfg["margin"]
    x1, y1, x2, y2 = x1 - mx, y1 - my, x2 + mx, y2 + my
    if not cfg["align"]:  # cách cũ: kẹp trong ảnh, không độn viền
        H, W = img.shape[:2]
        x1, y1, x2, y2 = max(0, np.floor(x1)), max(0, np.floor(y1)), min(W, np.ceil(x2)), min(H, np.ceil(y2))
    w, h = int(round(x2 - x1)), int(round(y2 - y1))
    if w < 8 or h < 8:
        return None
    M[:, 2] -= [x1, y1]
    crop = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    tm = np.zeros((h, w), np.uint8)
    cv2.fillPoly(tm, [np.round(cv2.transform(poly[None], M)[0]).astype(np.int32)], 1)
    if cfg["mask_outside"]:
        d = int(cfg["mask_dilate"])
        keep = cv2.dilate(tm, np.ones((2 * d + 1, 2 * d + 1), np.uint8)) if d > 0 else tm
        crop[keep == 0] = 0
    return crop, M, tm.astype(bool)


def to_crop(pts, M):
    return cv2.transform(np.asarray(pts, float)[None], M)[0]


def to_image(pts, M):
    return cv2.transform(np.asarray(pts, float)[None], cv2.invertAffineTransform(M))[0]
