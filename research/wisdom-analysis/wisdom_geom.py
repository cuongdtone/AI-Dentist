"""Phân tích hình học răng khôn từ mask răng có số FDI (đầu ra của model FDI-seg).

Quy ước ảnh panoramic (giống DENTEX): Q1 trên-trái ảnh, Q2 trên-phải, Q3 dưới-phải, Q4 dưới-trái.
Toạ độ ảnh: x sang phải, y xuống dưới.

Cho mỗi răng khôn (18/28/38/48), so với răng 7 cùng phần tư (M2):
  - Trục răng: PCA trên điểm ảnh của mask, định hướng về phía thân răng (crown).
  - Góc Winter: góc có dấu giữa trục M3 và trục M2, dương = nghiêng về phía gần (mesial).
      thẳng |a|<=10 | nghiêng gần 11..79 | nằm ngang 80..100 | nghiêng xa -11..-79 | khác (ngược, ...)
  - Độ sâu (Pell & Gregory A/B/C cho hàm dưới, Archer cho hàm trên): chiếu mask lên trục M2,
      r = (mặt nhai M2 - điểm cao nhất M3) / chiều dài M2.
      A: r <= depth_a (ngang/trên mặt nhai) | B: <= cej_frac (giữa mặt nhai và CEJ) | C: dưới CEJ.
      CEJ của M2 ước lượng = cej_frac * chiều dài răng tính từ mặt nhai (mask không có mốc CEJ).
  - Ngầm (impacted): luật r > impact_depth HOẶC |góc| > angle_thr (chỉnh bằng calibrate.py theo nhãn
      Impacted của DENTEX). Tách riêng với depth_a để lớp A/B/C giữ đúng định nghĩa hình học.
"""
import json, math, os
import numpy as np

DEFAULT_PARAMS = {
    "depth_a": 0.05,     # dung sai độ sâu cho loại A (theo tỉ lệ chiều dài M2)
    "cej_frac": 0.35,    # vị trí CEJ của M2 tính từ mặt nhai (tỉ lệ chiều dài răng)
    "impact_depth": 0.19,  # luật ngầm: depth_ratio vượt ngưỡng (chỉnh bằng calibrate.py)
    "angle_thr": 20.0,   # luật ngầm: |góc Winter| vượt ngưỡng (chỉnh bằng calibrate.py)
    "min_elong": 1.15,   # tỉ lệ trục dài/ngắn tối thiểu để tin trục PCA
}
WINTER_VI = {"vertical": "Thẳng", "mesioangular": "Nghiêng gần", "horizontal": "Nằm ngang",
             "distoangular": "Nghiêng xa", "other": "Khác"}
WISDOM = (18, 28, 38, 48)


def load_params(path=None):
    p = dict(DEFAULT_PARAMS)
    if path and os.path.exists(path):
        p.update(json.load(open(path)))
    return p


def directions(fdi):
    """(crown_dir, mesial_dir) dạng vector đơn vị trong toạ độ ảnh cho phần tư của răng."""
    q = fdi // 10
    crown = np.array([0.0, -1.0]) if q in (3, 4) else np.array([0.0, 1.0])  # hàm dưới: thân răng hướng lên
    mesial = np.array([1.0, 0.0]) if q in (1, 4) else np.array([-1.0, 0.0])  # nửa trái ảnh: phía gần = +x
    return crown, mesial


def mask_points(mask, step=2):
    ys, xs = np.nonzero(mask[::step, ::step])
    return np.stack([xs, ys], 1).astype(np.float64) * step


def tooth_axis(pts, fdi):
    """Trục dài (đơn vị, hướng về thân răng), tâm, độ thon dài (sqrt(l1/l2)), cách định hướng."""
    c = pts.mean(0)
    cov = np.cov((pts - c).T)
    w, v = np.linalg.eigh(cov)
    axis = v[:, 1]
    elong = math.sqrt(w[1] / max(w[0], 1e-9))
    crown, mesial = directions(fdi)
    # 1) thân răng rộng hơn chân răng: so độ rộng 30% hai đầu dọc trục
    s = (pts - c) @ axis
    perp = (pts - c) @ np.array([-axis[1], axis[0]])
    lo, hi = np.percentile(s, [30, 70])
    w_hi, w_lo = np.ptp(perp[s >= hi]) if (s >= hi).any() else 0, np.ptp(perp[s <= lo]) if (s <= lo).any() else 0
    how = "width"
    if max(w_hi, w_lo) > 1.15 * max(min(w_hi, w_lo), 1):
        if w_lo > w_hi:
            axis = -axis
    else:
        # 2) không rõ -> theo hướng thân răng kỳ vọng (ưu tiên nhẹ phía gần cho răng nằm ngang)
        how = "prior"
        if axis @ (crown + 0.15 * mesial) < 0:
            axis = -axis
    return axis, c, elong, how


def signed_angle(axis, fdi):
    """Góc (độ) của trục so với hướng thân răng kỳ vọng, dương = nghiêng về phía gần."""
    crown, mesial = directions(fdi)
    return math.degrees(math.atan2(axis @ mesial, axis @ crown))


def winter_class(a):
    if abs(a) <= 10:
        return "vertical"
    if 10 < a < 80:
        return "mesioangular"
    if 80 <= a <= 100:
        return "horizontal"
    if -80 < a < -10:
        return "distoangular"
    return "other"


def depth_class(r, p):
    if r is None:
        return None
    if r <= p["depth_a"]:
        return "A"
    return "B" if r <= p["cej_frac"] else "C"


def analyze_tooth(m3_mask, m2_mask, fdi, p=DEFAULT_PARAMS):
    """Trả về dict đặc trưng + kết luận cho 1 răng khôn. m2_mask có thể None."""
    out = {"fdi": fdi, "flags": []}
    pts3 = mask_points(m3_mask)
    if len(pts3) < 20:
        out["flags"].append("mask_m3_qua_nho"); return out
    ax3, c3, el3, how3 = tooth_axis(pts3, fdi)
    out.update(m3_center=c3.tolist(), m3_axis=ax3.tolist(), m3_elong=round(el3, 2), m3_orient=how3)
    if el3 < p["min_elong"]:
        out["flags"].append("truc_m3_khong_chac")  # vd mầm răng chưa có chân -> gần tròn
    a3 = signed_angle(ax3, fdi)

    if m2_mask is not None and len(pts2 := mask_points(m2_mask)) >= 20:
        ax2, c2, el2, _ = tooth_axis(pts2, fdi)
        ref = ax2 if el2 >= p["min_elong"] else directions(fdi)[0]
        if el2 < p["min_elong"]:
            out["flags"].append("truc_m2_khong_chac")
        a2 = signed_angle(ref, fdi)
        s2 = pts2 @ ref
        occ, apex = np.percentile(s2, 98), np.percentile(s2, 2)
        L = max(occ - apex, 1.0)
        top3 = np.percentile(pts3 @ ref, 98)
        r = float((occ - top3) / L)
        out.update(m2_center=c2.tolist(), m2_axis=ref.tolist(), m2_len=round(float(L), 1),
                   occ_level=float(occ), cej_level=float(occ - p["cej_frac"] * L), depth_ratio=round(r, 3))
    else:
        out["flags"].append("thieu_m2")
        a2, r = 0.0, None  # tham chiếu = trục đứng; không tính được độ sâu

    ang = (a3 - a2 + 180) % 360 - 180
    out["winter_angle"] = round(ang, 1)
    out["winter"] = winter_class(ang)
    out["depth"] = depth_class(r, p)
    out["impacted"] = bool((r is not None and r > p["impact_depth"]) or abs(ang) > p["angle_thr"])
    return out


def analyze_teeth(masks, p=DEFAULT_PARAMS, which=WISDOM):
    """masks: dict {fdi(int): bool mask HxW}. Trả về list kết quả cho các răng khôn có mặt."""
    return [analyze_tooth(masks[f], masks.get(f - 1), f, p) for f in which if f in masks]


def describe(r):
    """Chuỗi tiếng Việt ngắn gọn cho 1 kết quả."""
    if "winter" not in r:
        return f"{r['fdi']}: không phân tích được ({', '.join(r['flags'])})"
    s = f"{r['fdi']}: {WINTER_VI[r['winter']]} {r['winter_angle']:+.0f}°"
    if r["depth"]:
        s += f", độ sâu {r['depth']} (r={r['depth_ratio']:.2f})"
    s += ", NGẦM/LỆCH" if r["impacted"] else ", mọc bình thường"
    if r["flags"]:
        s += f" [{', '.join(r['flags'])}]"
    return s


# ---------------- tiện ích dùng chung ----------------
def predict_masks(model, img, imgsz=1024, conf=0.25, device=None, keep=(17, 18, 27, 28, 37, 38, 47, 48)):
    """Chạy model FDI-seg -> {fdi: mask} (nếu trùng số, giữ cái conf cao nhất)."""
    r = model.predict(img, imgsz=imgsz, conf=conf, retina_masks=True, device=device, verbose=False)[0]
    out, best = {}, {}
    if r.masks is None:
        return out
    ms = r.masks.data.cpu().numpy() > 0.5
    for m, c, s in zip(ms, r.boxes.cls.cpu().numpy(), r.boxes.conf.cpu().numpy()):
        f = int(model.names[int(c)])
        if f in keep and s > best.get(f, -1):
            best[f], out[f] = s, m
    return out


def draw(img, results, masks=None):
    """Vẽ trục M2/M3, mức mặt nhai/CEJ và nhãn (ASCII vì cv2 không vẽ được dấu tiếng Việt)."""
    import cv2
    vis = img.copy()
    short = {"vertical": "Vert", "mesioangular": "Mesio", "horizontal": "Horiz", "distoangular": "Disto", "other": "Other"}
    for r in results:
        if "winter" not in r:
            continue
        col = (0, 0, 255) if r["impacted"] else (0, 200, 0)
        if masks is not None and r["fdi"] in masks:
            cs, _ = cv2.findContours(masks[r["fdi"]].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cs, -1, col, 2)
        c, a = np.array(r["m3_center"]), np.array(r["m3_axis"])
        cv2.arrowedLine(vis, tuple((c - a * 120).astype(int)), tuple((c + a * 120).astype(int)), col, 3, tipLength=0.15)
        if "m2_center" in r:
            c2, a2 = np.array(r["m2_center"]), np.array(r["m2_axis"])
            cv2.arrowedLine(vis, tuple((c2 - a2 * 150).astype(int)), tuple((c2 + a2 * 150).astype(int)), (255, 180, 0), 2, tipLength=0.1)
            n = np.array([-a2[1], a2[0]])
            for lvl, lc in ((r["occ_level"], (255, 255, 0)), (r["cej_level"], (255, 0, 255))):
                base = c2 + a2 * (lvl - c2 @ a2)  # điểm trên trục M2 ở mức lvl
                cv2.line(vis, tuple((base - n * 250).astype(int)), tuple((base + n * 250).astype(int)), lc, 2)
        txt = f"{r['fdi']} {short[r['winter']]} {r['winter_angle']:+.0f}" + (f" {r['depth']}" if r["depth"] else "")
        org = (int(c[0]) - 80, int(c[1]) + (170 if r["fdi"] // 10 in (3, 4) else -150))
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 6)
        cv2.putText(vis, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 1.1, col, 2)
    return vis
