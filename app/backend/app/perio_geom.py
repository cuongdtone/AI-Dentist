"""Đo mức CEJ / đỉnh xương ổ răng / tiêu xương cho từng răng từ 3 VÙNG theo cả cung hàm
(model research/perio-seg hoặc nhãn bác sĩ final_data):
  crown_upper / crown_lower : vùng thân răng hàm trên / dưới -> mép phía chóp = đường CEJ
  bone_window               : vùng giữa đỉnh xương hàm trên và hàm dưới -> mép = đường đỉnh xương
Bản DUY NHẤT: research (e2e-evaluation) import trực tiếp từ đây.

Mỗi răng (polygon + số FDI từ bước phân đoạn):
  - Trục dài + hướng thân răng: wisdom_geom.tooth_axis.
  - 2 đường song song trục ở mặt GẦN / XA, cách trục side_frac x nửa bề rộng răng; đi từ đỉnh thân răng về chóp:
      CEJ   = điểm rời vùng thân răng (crown_*),
      crest = điểm rời bone_window (đi vào xương).
  - Chóp = đầu mask răng phía chân (trên trục giữa).
  - Tiêu xương (radiographic bone loss, tỉ lệ để không phụ thuộc độ phóng đại phim):
      r = (CEJ->crest) / (CEJ->chóp);  RBL = max(0, (r - phys_frac) / (1 - phys_frac))
      phys_frac ~ khoảng CEJ-đỉnh xương sinh lý (~1.5-2 mm) theo tỉ lệ chiều dài chân răng.
  - Giai đoạn (AAP/EFP 2017 theo % tiêu xương): <15% I · 15-33% II · >33% III/IV.
  - Tiêu xương dạng góc (dọc): |RBL gần - RBL xa| > angular_thr và RBL lớn nhất >= 15%.
"""
import cv2
import numpy as np
from .wisdom_geom import directions, tooth_axis

DEFAULT_PARAMS = {"side_frac": 0.6, "phys_frac": 0.1, "angular_thr": 0.15, "stage2": 0.15, "stage3": 0.33}
KINDS = ("crown_upper", "crown_lower", "bone_window")
STAGE_VI = {1: "I (<15%)", 2: "II (15–33%)", 3: "III/IV (>33%)"}


def _raster(polys, x0, y0, w, h):
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        if p is not None and len(p) >= 3:
            cv2.fillPoly(m, [np.round(np.asarray(p, float) - [x0, y0]).astype(np.int32)], 1)
    return m.astype(bool)


def _walk(line, tooth, crown, window):
    """line: Nx2 int toạ độ cục bộ đi từ phía thân răng về chóp. -> (i_cej, i_crest, i_start) hoặc None."""
    h, w = tooth.shape
    ok = (line[:, 0] >= 0) & (line[:, 0] < w) & (line[:, 1] >= 0) & (line[:, 1] < h)
    lx, ly = np.clip(line[:, 0], 0, w - 1), np.clip(line[:, 1], 0, h - 1)
    t, c, b = tooth[ly, lx] & ok, crown[ly, lx] & ok, window[ly, lx] & ok
    inside = np.flatnonzero(t)
    if len(inside) < 5:
        return None
    i0, i1 = inside[0], inside[-1]
    cej = crest = None
    seen_crown = False
    for i in range(i0, i1 + 1):
        if c[i]:
            seen_crown = True
        elif seen_crown and cej is None:
            cej = i
        if crest is None and not b[i]:
            crest = i
        if cej is not None and crest is not None:
            break
    return cej, crest, i0, i1


def measure_tooth(poly, fdi, crown_poly, window_poly, p=None):
    p = {**DEFAULT_PARAMS, **(p or {})}
    poly = np.asarray(poly, float)
    if len(poly) < 3 or crown_poly is None or window_poly is None:
        return None
    x1, y1 = poly.min(0); x2, y2 = poly.max(0)
    pad = 0.5 * max(x2 - x1, y2 - y1)
    x0, y0 = int(x1 - pad), int(y1 - pad)
    w, h = int(x2 + pad) - x0 + 1, int(y2 + pad) - y0 + 1
    tooth = _raster([poly], x0, y0, w, h)
    ys, xs = np.nonzero(tooth)
    if len(xs) < 50:
        return None
    pts = np.stack([xs, ys], 1).astype(float)
    axis, c, elong, _ = tooth_axis(pts[:: max(1, len(pts) // 4000)], int(fdi))
    n = np.array([-axis[1], axis[0]])
    _, mesial = directions(int(fdi))
    m_sign = 1.0 if n @ mesial >= 0 else -1.0
    u, v = (pts - c) @ axis, (pts - c) @ n
    u_top, u_apex = u.max(), np.percentile(u, 0.5)
    hw = np.percentile(np.abs(v), 95)
    crown, window = _raster([crown_poly], x0, y0, w, h), _raster([window_poly], x0, y0, w, h)
    us = np.arange(u_top + 3, u_apex - 3, -1.0)
    L_root = None
    out = {"fdi": str(fdi), "flags": []}
    if elong < 1.15:
        out["flags"].append("truc_khong_chac")
    for side, sgn in (("mesial", m_sign), ("distal", -m_sign)):
        off = sgn * p["side_frac"] * hw
        line_f = c + np.outer(us, axis) + off * n
        r = _walk(np.round(line_f).astype(int), tooth, crown, window)
        if r is None or r[0] is None:
            out[side] = None
            continue
        i_cej, i_crest, i0, i1 = r
        u_cej = us[i_cej]
        L = u_cej - u_apex
        if L <= 5:
            out[side] = None
            continue
        L_root = L if L_root is None else max(L_root, L)
        beyond = i_crest is None  # không gặp xương trong răng -> tiêu tới chóp
        u_crest = u_apex if beyond else us[i_crest]
        d = u_cej - u_crest
        ratio = float(d / L)
        rbl = float(np.clip((ratio - p["phys_frac"]) / (1 - p["phys_frac"]), 0, 1))
        to_img = lambda uu: [round(float(x), 1) for x in (c + axis * uu + off * n + [x0, y0])]
        out[side] = {"cej": to_img(u_cej), "crest": to_img(u_crest), "ratio": round(ratio, 3), "rbl": round(rbl, 3)}
        if beyond:
            out["flags"].append(f"{side}_khong_thay_xuong")
    sides = [out[s] for s in ("mesial", "distal") if out.get(s)]
    if not sides:
        return None
    out["apex"] = [round(float(x), 1) for x in (c + axis * u_apex + [x0, y0])]
    rbl = max(s["rbl"] for s in sides)
    out["rbl"] = round(rbl, 3)
    out["stage"] = 1 if rbl < p["stage2"] else 2 if rbl <= p["stage3"] else 3
    out["angular"] = bool(len(sides) == 2 and abs(sides[0]["rbl"] - sides[1]["rbl"]) > p["angular_thr"] and rbl >= p["stage2"])
    return out


def regions_by_kind(regions):
    """list {kind, conf, polygon} -> {kind: polygon} (mỗi loại lấy vùng conf cao nhất)."""
    best = {}
    for r in regions or []:
        if r.get("kind") in KINDS and len(r.get("polygon", [])) >= 3 and r.get("conf", 1) > best.get(r["kind"], {}).get("conf", -1):
            best[r["kind"]] = r
    return {k: v["polygon"] for k, v in best.items()}


def measure(teeth, regions, p=None):
    """teeth: list {id, fdi, polygon}; regions: list {kind, polygon,...} -> list kết quả từng răng (sắp theo FDI)."""
    R = regions_by_kind(regions)
    out = []
    for t in teeth:
        f = str(t["fdi"])
        crown = R.get("crown_upper" if f[0] in "12" else "crown_lower")
        r = measure_tooth(t["polygon"], int(f), crown, R.get("bone_window"), p)
        if r:
            r["tooth_id"] = t.get("id")
            out.append(r)
    out.sort(key=lambda r: r["fdi"])
    return out


def describe(r):
    s = f"{r['fdi']}: tiêu xương {r['rbl'] * 100:.0f}% · giai đoạn {STAGE_VI[r['stage']]}"
    sides = [f"{n} {r[k]['rbl'] * 100:.0f}%" for k, n in (("mesial", "gần"), ("distal", "xa")) if r.get(k)]
    if sides:
        s += f" ({', '.join(sides)})"
    if r["angular"]:
        s += ", dạng góc (dọc)"
    return s
