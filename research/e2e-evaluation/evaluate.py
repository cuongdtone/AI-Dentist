"""Đánh giá end-to-end pipeline của webapp trên final_data: ảnh -> phân đoạn răng -> răng khôn -> sâu răng.
Chạy ĐÚNG code backend (app/backend/app: ToothSegmenter ONNX, analyze_wisdom, analyze_caries) với config mặc định.

Model: luôn lấy best.pt MỚI NHẤT (theo mtime) trong research/tooth-detect/runs, research/caries-seg/runs và
research/perio-seg/runs (CEJ + mức xương; chưa có run -> bỏ qua phần nha chu),
export sang ONNX (cache theo tên run + mtime trong cache/onnx) -> giống hệt cách app chạy.

Dùng:
  P=../../wisdom_tooth/env/bin/python
  $P evaluate.py                         # toàn bộ final_data, ảnh combine + report
  $P evaluate.py --limit 20 --no-cache   # thử nhanh
  $P evaluate.py --tooth-weights X.pt --caries-weights Y.pt

Ra: out/<tooth_run>__<caries_run>/
  vis/<split>/<doc>_<ảnh>.jpg   trái: nhãn bác sĩ (sâu răng, răng khôn lệch ngầm) | phải: dự đoán (đồ hoạ như webapp)
  report.md, report.json, wisdom_teeth.csv, caries_teeth.csv, caries_lesions.csv

Nhãn final_data (Supervisely): 'Sau rang (polygon)' = vùng sâu; 'Rang khon ham duoi lech ngam' = box (chỉ hàm dưới);
'Truc rang' (polyline khép kín) = viền răng (không có số FDI, chỉ để vẽ); 'Muc CEJ' (2 vòng = vùng thân răng trên/dưới)
+ 'Muc xuong nha chu' (1 vòng = vùng giữa đỉnh xương 2 hàm). Ảnh trùng hash chỉ tính 1 lần.
Nha chu: đo CEJ / đỉnh xương / % tiêu xương bằng app/backend/app/perio_geom.py trên CÙNG polygon răng của app cho cả
vùng bác sĩ (GT) và vùng model -> sai số chỉ do model vùng.
Split train/val/test theo wisdom_tooth/data (test không dùng để train model răng lẫn model sâu răng) -> số liệu chính = test.
"""
import argparse, colorsys, csv, glob, hashlib, json, math, os, re, shutil, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BACKEND = ROOT / "app/backend"
os.environ.setdefault("DATA_DIR", str(HERE / "cache/appdata"))  # app.db (import pipeline -> db) không lẫn vào app thật
sys.path.insert(0, str(BACKEND))
from app import perio_geom, pipeline  # noqa: E402
from app.caries import load_crop_cfg  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
CARIES_CLS, WISDOM_CLS, CONTOUR_CLS = "Sau rang (polygon)", "Rang khon ham duoi lech ngam", "Truc rang"
CEJ_CLS, BONE_CLS = "Muc CEJ", "Muc xuong nha chu"
SPLITS = ("test", "val", "train")
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


# ---------------------------------------------------------------- model
def latest_best(pattern, required=True):
    c = glob.glob(str(ROOT / pattern), recursive=True)
    if not c:
        if not required:
            return None
        sys.exit(f"Không thấy weights: {pattern}")
    return Path(max(c, key=os.path.getmtime))


def run_arg(weights, key):
    args = Path(weights).resolve().parents[1] / "args.yaml"
    m = re.search(rf"^{key}:\s*(.+)$", args.read_text(), re.M) if args.exists() else None
    return m.group(1).strip().strip("'\"") if m else None


def export_onnx(weights, kind):
    """.pt -> cache/onnx/<run>_<mtime>.onnx (export 1 lần). caries: chép crop.json của dataset thành <onnx>.json."""
    weights = Path(weights)
    run = weights.resolve().parents[1].name
    imgsz = int(run_arg(weights, "imgsz") or (384 if kind == "caries" else 1024))
    out = HERE / "cache/onnx" / f"{kind}_{run}_{int(weights.stat().st_mtime)}_{imgsz}.onnx"
    if not out.exists():
        from ultralytics import YOLO
        print(f"export {weights} -> {out.name} (imgsz {imgsz})", flush=True)
        src = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=17, simplify=True, dynamic=False, device="cpu")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(src, out)
    if kind == "caries":
        data = run_arg(weights, "data")
        crop = (ROOT / "research/caries-seg" / data).parent / "crop.json" if data else None
        side = out.with_suffix(".json")
        if crop and crop.exists():
            shutil.copy2(crop, side)
        else:
            side.unlink(missing_ok=True)
    return out, run


# ---------------------------------------------------------------- dữ liệu
def split_names():
    out = {}
    for s in SPLITS:
        for f in os.listdir(ROOT / "wisdom_tooth/data/images" / s):
            out[os.path.splitext(f)[0]] = s
    return out


def collect(src):
    split, seen, items = split_names(), set(), []
    for doc in sorted(os.listdir(src)):
        d = Path(src) / doc
        if not (d / "ann").is_dir():
            continue
        for f in sorted(os.listdir(d / "ann")):
            name = f[:-5]
            info = d / "img_info" / f
            h = json.load(open(info, encoding="utf8")).get("hash") if info.exists() else f"{doc}/{name}"
            if h in seen:
                continue
            seen.add(h)
            ann = json.load(open(d / "ann" / f, encoding="utf8"))
            car, box, cont, cej, bone = [], [], [], [], []
            for o in ann["objects"]:
                pts = o["points"]["exterior"]
                if o["classTitle"] == CARIES_CLS and len(pts) >= 3:
                    car.append([[float(x), float(y)] for x, y in pts])
                elif o["classTitle"] == WISDOM_CLS:
                    (x1, y1), (x2, y2) = pts[:2]
                    box.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
                elif o["classTitle"] == CONTOUR_CLS and len(pts) >= 3:
                    cont.append(pts)
                elif o["classTitle"] == CEJ_CLS and len(pts) >= 3:
                    cej.append(pts)
                elif o["classTitle"] == BONE_CLS and len(pts) >= 3:
                    bone.append(pts)
            regions = None  # như research/perio-seg/prepare_data.py: cần đúng 2 vòng CEJ + 1 vòng xương
            if len(cej) == 2 and len(bone) == 1:
                up, lo = sorted(cej, key=lambda p: np.mean([q[1] for q in p]))
                regions = [{"id": i, "kind": k, "conf": 1.0, "polygon": [[float(x), float(y)] for x, y in q]}
                           for i, (k, q) in enumerate(zip(perio_geom.KINDS, (up, lo, bone[0])))]
            stem = f"{doc}_{os.path.splitext(name)[0]}"
            items.append(dict(doc=doc, name=name, stem=stem, split=split.get(stem, "unknown"), img=str(d / "img" / name),
                              caries=car, boxes=box, contours=cont, regions=regions))
    return items


# ---------------------------------------------------------------- hình học
def poly_mask(polys, x0, y0, w, h):
    m = np.zeros((h, w), np.uint8)
    for p in polys:
        cv2.fillPoly(m, [np.round(np.asarray(p) - [x0, y0]).astype(np.int32)], 1)
    return m.astype(bool)


def bbox(p):
    a = np.asarray(p)
    return a[:, 0].min(), a[:, 1].min(), a[:, 0].max(), a[:, 1].max()


def poly_inter(a, b):
    """-> (diện tích giao, diện tích a, diện tích b) theo pixel, tính trong bbox hợp."""
    ba, bb = bbox(a), bbox(b)
    if ba[2] < bb[0] or bb[2] < ba[0] or ba[3] < bb[1] or bb[3] < ba[1]:
        return 0, None, None
    x0, y0 = int(min(ba[0], bb[0])) - 1, int(min(ba[1], bb[1])) - 1
    w, h = int(max(ba[2], bb[2])) - x0 + 2, int(max(ba[3], bb[3])) - y0 + 2
    ma, mb = poly_mask([a], x0, y0, w, h), poly_mask([b], x0, y0, w, h)
    return int((ma & mb).sum()), int(ma.sum()), int(mb.sum())


def iou(a, b):
    i, sa, sb = poly_inter(a, b)
    return i / max(sa + sb - i, 1) if i else 0.0


def box_iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1)


# ---------------------------------------------------------------- đánh giá 1 ảnh
def eval_image(it, pred, a):
    teeth, wis, car = pred["teeth"], pred["wisdom"], pred["caries"]
    by_id = {t["id"]: t for t in teeth}
    r = dict(wisdom=[], caries_teeth=[], lesions=[], wis_box_hit=0, car_unassigned=0)

    # Răng khôn hàm dưới (nhãn chỉ có 38/48): mỗi răng app phân tích được -> label = khớp box bác sĩ
    hit = set()
    for w in wis:
        if w["fdi"] not in ("38", "48") or not w.get("winter") or w.get("tooth_id") not in by_id:
            continue
        poly = np.asarray(by_id[w["tooth_id"]]["polygon"])
        mb = bbox(poly); cx, cy = poly.mean(0)
        lab = 0
        for j, b in enumerate(it["boxes"]):
            if box_iou(mb, b) >= a.wis_iou or (b[0] <= cx <= b[2] and b[1] <= cy <= b[3]):
                lab = 1; hit.add(j); break
        r["wisdom"].append(dict(fdi=w["fdi"], label=lab, pred=int(w["impacted"]), angle=w["winter_angle"],
                                winter=w["winter"], depth=w.get("depth") or "", flags="|".join(w.get("flags", []))))
    r["wis_box_hit"] = len(hit)
    r["wis_box_hit_ids"] = sorted(hit)

    # Sâu răng mức vùng: ghép GT <-> dự đoán tham lam theo IoU (>= --car-iou)
    pairs = sorted(((iou(g, p["polygon"]), gi, pi) for gi, g in enumerate(it["caries"]) for pi, p in enumerate(car)), reverse=True)
    gm, pm = {}, {}
    for v, gi, pi in pairs:
        if v < a.car_iou or gi in gm or pi in pm:
            continue
        gm[gi], pm[pi] = (pi, v), (gi, v)
    for gi in range(len(it["caries"])):
        r["lesions"].append(dict(kind="gt", matched=int(gi in gm), iou=round(gm[gi][1], 3) if gi in gm else 0, conf=""))
    for pi, p in enumerate(car):
        r["lesions"].append(dict(kind="pred", matched=int(pi in pm), iou=round(pm[pi][1], 3) if pi in pm else 0, conf=p["conf"]))
    r["gt_matched"], r["pred_matched"] = set(gm), set(pm)

    # Sâu răng mức răng: mỗi vùng GT gán cho răng (app tìm thấy) giao nhiều nhất
    gt_teeth = set()
    for g in it["caries"]:
        best = max(((poly_inter(g, t["polygon"])[0], t["id"]) for t in teeth), default=(0, None))
        if best[0] > 0:
            gt_teeth.add(best[1])
        else:
            r["car_unassigned"] += 1
    pred_teeth = defaultdict(float)
    for c in car:
        pred_teeth[c["tooth_id"]] = max(pred_teeth[c["tooth_id"]], c["conf"])
    for t in teeth:
        r["caries_teeth"].append(dict(fdi=t["fdi"], label=int(t["id"] in gt_teeth), pred=int(t["id"] in pred_teeth),
                                      conf=round(pred_teeth.get(t["id"], 0), 3)))
    r["gt_caries_teeth"] = gt_teeth
    return r


def region_iou(a, b, W, H, f=0.25):
    """IoU 2 polygon trên lưới thu nhỏ (nhanh, đủ chính xác cho vùng lớn)."""
    w, h = max(1, int(W * f)), max(1, int(H * f))
    ma, mb = poly_mask([np.asarray(a) * f], 0, 0, w, h), poly_mask([np.asarray(b) * f], 0, 0, w, h)
    return float((ma & mb).sum() / max((ma | mb).sum(), 1))


def eval_perio(it, pred, cfg, W, H):
    """GT = perio_geom trên vùng bác sĩ + polygon răng của app; so với kết quả app (vùng model)."""
    if not it["regions"] or "perio" not in pred:
        return None
    gt = {m["tooth_id"]: m for m in perio_geom.measure(pred["teeth"], it["regions"], cfg)}
    pm = {m["tooth_id"]: m for m in pred["perio"]}
    P = perio_geom.regions_by_kind(pred["regions"])
    G = perio_geom.regions_by_kind(it["regions"])
    ious = {k: (region_iou(G[k], P[k], W, H) if k in P else 0.0) for k in perio_geom.KINDS}
    teeth, sides = [], []
    for tid, g in gt.items():
        p = pm.get(tid)
        teeth.append(dict(fdi=g["fdi"], gt_rbl=g["rbl"], gt_stage=g["stage"], pred_rbl=p["rbl"] if p else "", pred_stage=p["stage"] if p else ""))
        if not p:
            continue
        for s in ("mesial", "distal"):
            if g.get(s) and p.get(s):
                d = lambda k: float(np.hypot(*(np.asarray(g[s][k]) - p[s][k])))
                sides.append(dict(fdi=g["fdi"], side=s, cej_err=round(d("cej"), 1), crest_err=round(d("crest"), 1),
                                  gt_rbl=g[s]["rbl"], pred_rbl=p[s]["rbl"]))
    return dict(iou=ious, teeth=teeth, sides=sides, gt=list(gt.values()))


def kappa_w(y, yh, k=3):
    """Cohen's kappa trọng số bậc 2 cho giai đoạn 1..k."""
    y, yh = np.asarray(y, int) - 1, np.asarray(yh, int) - 1
    if not len(y):
        return 0.0
    O = np.zeros((k, k))
    for a, b in zip(y, yh):
        O[a, b] += 1
    E = np.outer(O.sum(1), O.sum(0)) / O.sum()
    Wt = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    return float(1 - (Wt * O).sum() / max((Wt * E).sum(), 1e-9))


def perio_stats(S):
    S = [r["perio"] for r in S if r.get("perio")]
    if not S:
        return None
    ious = {k: round(float(np.mean([r["iou"][k] for r in S])), 4) for k in perio_geom.KINDS}
    sides = [x for r in S for x in r["sides"]]
    teeth = [t for r in S for t in r["teeth"]]
    both = [t for t in teeth if t["pred_stage"] != ""]
    pct = lambda a, q: round(float(np.percentile(a, q)), 1) if len(a) else None
    ce, cr = [x["cej_err"] for x in sides], [x["crest_err"] for x in sides]
    return dict(
        images=len(S), region_iou=ious, teeth_gt=len(teeth), teeth_both=len(both), sides=len(sides),
        cej_err_px=dict(median=pct(ce, 50), mean=round(float(np.mean(ce)), 1) if ce else None, p90=pct(ce, 90)),
        crest_err_px=dict(median=pct(cr, 50), mean=round(float(np.mean(cr)), 1) if cr else None, p90=pct(cr, 90)),
        rbl_mae=round(float(np.mean([abs(t["gt_rbl"] - t["pred_rbl"]) for t in both])), 4) if both else None,
        stage_acc=round(float(np.mean([t["gt_stage"] == t["pred_stage"] for t in both])), 4) if both else None,
        stage_kappa=round(kappa_w([t["gt_stage"] for t in both], [t["pred_stage"] for t in both]), 4) if both else None,
        stage_confusion=[[sum(1 for t in both if t["gt_stage"] == a and t["pred_stage"] == b) for b in (1, 2, 3)] for a in (1, 2, 3)],
        stage2plus=prf([int(t["gt_stage"] >= 2) for t in both], [int(t["pred_stage"] >= 2) for t in both]),
    )


def prf(y, yh):
    y, yh = np.asarray(y, int), np.asarray(yh, int)
    tp = int(((y == 1) & (yh == 1)).sum()); fp = int(((y == 0) & (yh == 1)).sum())
    fn = int(((y == 1) & (yh == 0)).sum()); tn = int(((y == 0) & (yh == 0)).sum())
    P, R = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    return dict(n=len(y), tp=tp, fp=fp, fn=fn, tn=tn, precision=round(P, 4), recall=round(R, 4),
                f1=round(2 * P * R / max(P + R, 1e-9), 4), specificity=round(tn / max(tn + fp, 1), 4),
                accuracy=round(float((y == yh).mean()) if len(y) else 0, 4))


# ---------------------------------------------------------------- vẽ (đồ hoạ như webapp)
QUAD_HUE = {"1": 213, "2": 18, "3": 158, "4": 330}
RED, GREEN, BLUE, YELLOW, MAGENTA = (68, 68, 239), (94, 197, 34), (250, 165, 96), (21, 204, 250), (239, 70, 217)  # BGR
DARK_RED = (28, 28, 185)


def tooth_color(fdi):
    r, g, b = colorsys.hls_to_rgb(QUAD_HUE.get(fdi[0], 0) / 360, 0.6, 0.85)
    return int(b * 255), int(g * 255), int(r * 255)


def pts(p):
    return np.round(np.asarray(p)).astype(np.int32).reshape(-1, 1, 2)


def blend(dst, layer, mask, alpha):
    dst[mask] = (dst[mask] * (1 - alpha) + layer[mask] * alpha).astype(np.uint8)


class Painter:
    """Vẽ hình bằng cv2 (khử răng cưa), chữ bằng PIL (có dấu tiếng Việt) - gom chữ lại vẽ một lần cuối."""

    def __init__(self, img, scale):
        self.img, self.s, self.texts = img, scale, []
        self.lw = max(2, round(scale * 2))

    def poly(self, p, color, w=1.0, alpha=1.0, fill=None, fill_alpha=0.0):
        if fill is not None and fill_alpha > 0:
            layer = self.img.copy(); cv2.fillPoly(layer, [pts(p)], fill)
            m = np.zeros(self.img.shape[:2], np.uint8); cv2.fillPoly(m, [pts(p)], 1)
            blend(self.img, layer, m.astype(bool), fill_alpha)
        if alpha >= 1:
            cv2.polylines(self.img, [pts(p)], True, color, max(1, round(self.lw * w)), cv2.LINE_AA)
        else:
            layer = self.img.copy(); cv2.polylines(layer, [pts(p)], True, color, max(1, round(self.lw * w)), cv2.LINE_AA)
            cv2.addWeighted(layer, alpha, self.img, 1 - alpha, 0, self.img)

    def line(self, a, b, color, w=1.0, arrow=False, dash=None):
        a, b = np.asarray(a, float), np.asarray(b, float)
        t = max(1, round(self.lw * w))
        if dash:
            L = np.linalg.norm(b - a); d = (b - a) / max(L, 1e-6); on, off = dash[0] * self.s, dash[1] * self.s
            x = 0.0
            while x < L:
                cv2.line(self.img, tuple(np.round(a + d * x).astype(int)), tuple(np.round(a + d * min(x + on, L)).astype(int)), color, t, cv2.LINE_AA)
                x += on + off
        elif arrow:
            cv2.arrowedLine(self.img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, t, cv2.LINE_AA,
                            tipLength=min(0.3, 14 * self.s / max(np.linalg.norm(b - a), 1)))
        else:
            cv2.line(self.img, tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int)), color, t, cv2.LINE_AA)

    def text(self, xy, s, size, color=(255, 255, 255), anchor="mm"):
        self.texts.append((xy, s, size, color, anchor))

    def flush(self):
        im = Image.fromarray(self.img[:, :, ::-1])
        d = ImageDraw.Draw(im)
        for xy, s, size, color, anchor in self.texts:
            f = ImageFont.truetype(FONT, max(8, round(size * self.s)))
            d.text(xy, s, font=f, fill=color[::-1], anchor=anchor, stroke_width=max(1, round(1.5 * self.s)), stroke_fill=(0, 0, 0))
        self.img[:] = np.asarray(im)[:, :, ::-1]


STAGE_BGR = {1: (94, 197, 34), 2: (11, 158, 245), 3: (68, 68, 239)}
CEJ_BGR, CREST_BGR = (239, 70, 217), (22, 115, 249)


def draw_perio(P, regions, meas, scale):
    """Như webapp: viền vùng thân răng (tím, đứt nét) + vùng xương (cam); đoạn CEJ->đỉnh xương màu theo giai đoạn, % cho GĐ >= II."""
    for r in regions or []:
        p = np.asarray(r["polygon"], float)
        if r["kind"] == "bone_window":
            P.poly(p, CREST_BGR, 0.8, alpha=0.85)
        else:  # đứt nét
            for a, b in zip(p, np.roll(p, -1, 0)):
                P.line(a, b, CEJ_BGR, 0.8, dash=(6, 3))
    for m in meas or []:
        S = [m[k] for k in ("mesial", "distal") if m.get(k)]
        col = STAGE_BGR[m["stage"]]
        for s in S:
            P.line(s["cej"], s["crest"], col, 1.5)
            cv2.circle(P.img, tuple(np.round(s["cej"]).astype(int)), max(2, round(3 * scale)), CEJ_BGR, -1, cv2.LINE_AA)
            cv2.circle(P.img, tuple(np.round(s["crest"]).astype(int)), max(2, round(3 * scale)), CREST_BGR, -1, cv2.LINE_AA)
        if m["stage"] >= 2 and S:
            mx, my = np.mean([s["crest"] for s in S], 0)
            d = np.asarray(m["apex"]) - [mx, my]; d = d / max(np.linalg.norm(d), 1)
            P.text((mx + d[0] * 18 * scale, my + d[1] * 18 * scale), f"{round(m['rbl'] * 100)}%{'∠' if m['angular'] else ''}", 12, col)


def draw_pred(img, pred, scale, hl=None):
    """Như ImageViewer: polygon răng màu theo cung hàm + FDI, vùng sâu viền đỏ + conf, trục răng khôn."""
    out = img.copy(); P = Painter(out, scale)
    draw_perio(P, pred.get("regions"), pred.get("perio"), scale)
    for t in pred["teeth"]:
        c = tooth_color(t["fdi"])
        P.poly(t["polygon"], c, 1.0, alpha=0.75)
        cx, cy = np.asarray(t["polygon"]).mean(0)
        P.text((cx, cy), t["fdi"], 16)
    for i, c in enumerate(pred["caries"]):
        miss = hl is not None and i not in hl  # không khớp nhãn nào (FP) -> giữ đồ hoạ app, chỉ thêm ✗ vào nhãn conf
        P.poly(c["polygon"], RED, 1.0)
        top = min(c["polygon"], key=lambda p: p[1])
        P.text((top[0], top[1] - 9 * scale), f"{c['conf'] * 100:.0f}%" + (" ✗" if miss else ""), 15, RED)
    for w in pred["wisdom"]:
        if not w.get("m3_axis"):
            continue
        col = RED if w["impacted"] else GREEN
        cx, cy = w["m3_center"]; ax, ay = w["m3_axis"]
        L3 = w["m2_len"] / 2 if w.get("m2_len") else 120
        P.line((cx - ax * L3, cy - ay * L3), (cx + ax * L3, cy + ay * L3), col, 1.6, arrow=True)
        if w.get("m2_axis"):
            bx, by = w["m2_center"]; ux, uy = w["m2_axis"]; h2 = w["m2_len"] / 2
            P.line((bx - ux * h2, by - uy * h2), (bx + ux * h2, by + uy * h2), BLUE, 1.0, dash=(6, 4))
            n, half = (-uy, ux), w["m2_len"] * 0.9
            for lvl, cc in ((w.get("occ_level"), YELLOW), (w.get("cej_level"), MAGENTA)):
                if lvl is None:
                    continue
                t = lvl - (bx * ux + by * uy); ox, oy = bx + ux * t, by + uy * t
                P.line((ox - n[0] * half, oy - n[1] * half), (ox + n[0] * half, oy + n[1] * half), cc, 1.0)
        below = w["fdi"][0] in "34"
        P.text((cx, cy + (1 if below else -1) * (L3 + 22 * scale)),
               f"{w['fdi']} {'+' if w['winter_angle'] > 0 else ''}{round(w['winter_angle'])}°{' ' + w['depth'] if w.get('depth') else ''}", 17, col)
    P.flush()
    return out


def draw_gt(img, it, scale, gt_perio=None):
    out = img.copy(); P = Painter(out, scale)
    draw_perio(P, it["regions"], gt_perio, scale)
    for c in it["contours"]:
        P.poly(c, (210, 210, 210), 0.8, alpha=0.45)
    for c in it["caries"]:
        P.poly(c, RED, 1.0)
    for x1, y1, x2, y2 in it["boxes"]:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), RED, P.lw * 2, cv2.LINE_AA)
        P.text((x1 + 4 * scale, y1 - 6 * scale), "Răng khôn lệch ngầm", 17, RED, anchor="lb")
    P.flush()
    return out


def header(w, lines, scale, color=(30, 30, 30)):
    h = round(34 * scale) * len(lines) + round(12 * scale)
    bar = np.full((h, w, 3), color, np.uint8)
    im = Image.fromarray(bar); d = ImageDraw.Draw(im)
    f = ImageFont.truetype(FONT, max(10, round(22 * scale)))
    for i, s in enumerate(lines):
        d.text((round(14 * scale), round(8 * scale) + i * round(34 * scale)), s, font=f, fill=(255, 255, 255))
    return np.asarray(im)


def combine(img, it, pred, ev, a):
    H, W = img.shape[:2]
    scale = max(1.0, W / 1400)  # nét / chữ theo cỡ ảnh (webapp vẽ theo pixel màn hình)
    pe = ev.get("perio")
    left, right = draw_gt(img, it, scale, pe["gt"] if pe else None), draw_pred(img, pred, scale, ev["pred_matched"])
    n2 = lambda L: sum(1 for m in L if m["stage"] >= 2)
    nt = len({c["tooth_id"] for c in pred["caries"]})
    imp = [w["fdi"] for w in pred["wisdom"] if w.get("impacted")]
    tp = len(ev["gt_matched"])
    hl = [f"NHÃN BÁC SĨ · {it['split']} · {it['doc']}/{it['name']}",
          f"Sâu răng: {len(it['caries'])} vùng · Răng khôn lệch ngầm: {len(it['boxes'])} box"
          + (f" · Tiêu xương ≥ II: {n2(pe['gt'])} răng" if pe else " · Không có nhãn CEJ/xương" if "perio" in pred else "")]
    hr = [f"DỰ ĐOÁN · {len(pred['teeth'])} răng",
          f"Sâu răng: {len(pred['caries'])} vùng / {nt} răng (khớp {tp}/{len(it['caries'])} nhãn, IoU≥{a.car_iou}) · "
          f"Ngầm/lệch: {', '.join(imp) if imp else 'không'}"
          + (f" · Tiêu xương ≥ II: {n2(pred['perio'])} răng" if "perio" in pred else "")]
    sep = np.full((H, max(4, round(6 * scale)), 3), 255, np.uint8)
    hdr = np.hstack([header(W, hl, scale), np.full((round(34 * scale) * 2 + round(12 * scale), sep.shape[1], 3), 255, np.uint8), header(W, hr, scale)])
    out = np.vstack([hdr, np.hstack([left, sep, right])])
    if a.vis_scale != 1:
        out = cv2.resize(out, None, fx=a.vis_scale, fy=a.vis_scale, interpolation=cv2.INTER_AREA)
    return out


# ---------------------------------------------------------------- báo cáo
def summarize(rows_by_split, extra):
    rep = {}
    for name, subset in rows_by_split.items():
        R = subset
        wis = [w for r in R for w in r["wisdom"]]
        nbox = sum(r["n_box"] for r in R); nbox_hit = sum(r["wis_box_hit"] for r in R)
        lab = [r for r in R if r["n_gt_caries"] > 0]  # ảnh bác sĩ có vẽ sâu răng
        def car_stats(S):
            gt = sum(r["n_gt_caries"] for r in S); pr = sum(r["n_pred_caries"] for r in S)
            mg = sum(len(r["gt_matched"]) for r in S); mp = sum(len(r["pred_matched"]) for r in S)
            Pp, Rr = mp / max(pr, 1), mg / max(gt, 1)
            ct = [c for r in S for c in r["caries_teeth"]]
            img_y = [int(r["n_gt_caries"] > 0) for r in S]; img_p = [int(r["n_pred_caries"] > 0) for r in S]
            return dict(images=len(S), gt_lesions=gt, pred_lesions=pr, lesion=dict(precision=round(Pp, 4), recall=round(Rr, 4),
                        f1=round(2 * Pp * Rr / max(Pp + Rr, 1e-9), 4), tp_gt=mg, tp_pred=mp),
                        gt_lesions_no_tooth=sum(r["car_unassigned"] for r in S),
                        tooth=prf([c["label"] for c in ct], [c["pred"] for c in ct]), image=prf(img_y, img_p))
        rep[name] = dict(
            images=len(R),
            wisdom=dict(lower_teeth=prf([w["label"] for w in wis], [w["pred"] for w in wis]),
                        by_fdi={f: prf([w["label"] for w in wis if w["fdi"] == f], [w["pred"] for w in wis if w["fdi"] == f]) for f in ("38", "48")},
                        gt_boxes=nbox, gt_boxes_matched_tooth=nbox_hit,
                        gt_boxes_detected_impacted=sum(r["wis_box_tp"] for r in R)),
            caries_labeled_images=car_stats(lab),
            caries_all_images=car_stats(R),
            perio=perio_stats(R),
        )
    rep.update(extra)
    return rep


def fmt(m):
    return f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} | {m['specificity']:.3f} | {m['accuracy']:.3f} | {m['tp']} / {m['fp']} / {m['fn']} / {m['tn']}"


def write_md(rep, path, a):
    L = [f"# Đánh giá end-to-end — {rep['tag']}", "",
         f"- Ngày chạy: {rep['date']} · ảnh (bỏ trùng hash): {rep['n_images']} · thời gian: {rep['seconds']:.0f}s",
         f"- Model răng: `{rep['tooth_weights']}`", f"- Model sâu răng: `{rep['caries_weights']}` (crop: {rep['crop_cfg']})",
         "- Pipeline: code backend của app, config mặc định (seg conf 0.25, caries conf 0.25, wisdom params mặc định).",
         "- **Số liệu chính: split `test`** (không dùng để train model răng lẫn model sâu răng). `val`/`train` đã thấy lúc train → lạc quan.", ""]
    H = "| Split | Precision | Recall | F1 | Specificity | Accuracy | TP / FP / FN / TN |\n|---|---|---|---|---|---|---|"
    splits = [s for s in (*SPLITS, "all") if s in rep]
    L += ["## Răng khôn ngầm/lệch (hàm dưới 38/48)", "",
          "Đơn vị: răng 38/48 app tìm thấy và phân tích được. Nhãn = 1 nếu mask khớp box *Rang khon ham duoi lech ngam* "
          f"(IoU ≥ {a.wis_iou} hoặc tâm mask nằm trong box). Giả định bác sĩ khoanh đủ mọi răng khôn hàm dưới lệch ngầm.", "", H]
    L += [f"| {s} | {fmt(rep[s]['wisdom']['lower_teeth'])} |" for s in splits]
    L += ["", "| Split | Box bác sĩ | Có răng app khớp | App kết luận ngầm/lệch (recall theo box) |", "|---|---|---|---|"]
    for s in splits:
        w = rep[s]["wisdom"]
        L.append(f"| {s} | {w['gt_boxes']} | {w['gt_boxes_matched_tooth']} | {w['gt_boxes_detected_impacted']} ({w['gt_boxes_detected_impacted'] / max(w['gt_boxes'], 1):.1%}) |")
    L += ["", "Theo răng (test):", "", H]
    if "test" in rep:
        L += [f"| {f} | {fmt(m)} |" for f, m in rep["test"]["wisdom"]["by_fdi"].items()]
    for key, title, note in (
        ("caries_labeled_images", "Sâu răng — ảnh bác sĩ có vẽ sâu răng", "Chỉ ảnh có ≥ 1 polygon *Sau rang (polygon)* (giống tập train: ảnh không vẽ sâu răng có thể là chưa gán nhãn)."),
        ("caries_all_images", "Sâu răng — toàn bộ ảnh", "Coi ảnh không có polygon sâu răng là không sâu (bi quan cho precision nếu bác sĩ bỏ sót).")):
        L += ["", f"## {title}", "", note, "",
              f"**Mức răng** — đơn vị: răng app tìm thấy; nhãn = 1 nếu có vùng sâu bác sĩ giao nhiều nhất với răng đó; dự đoán = 1 nếu app báo ≥ 1 vùng sâu.", "", H]
        L += [f"| {s} | {fmt(rep[s][key]['tooth'])} |" for s in splits]
        L += ["", f"**Mức vùng** — ghép 1-1 tham lam theo IoU polygon ≥ {a.car_iou}.", "",
              "| Split | Ảnh | Vùng GT | Vùng dự đoán | Precision | Recall | F1 | GT không thuộc răng nào |", "|---|---|---|---|---|---|---|---|"]
        for s in splits:
            c = rep[s][key]; l = c["lesion"]
            L.append(f"| {s} | {c['images']} | {c['gt_lesions']} | {c['pred_lesions']} | {l['precision']:.3f} | {l['recall']:.3f} | {l['f1']:.3f} | {c['gt_lesions_no_tooth']} |")
        L += ["", "**Mức ảnh** — có / không có sâu răng.", "", H]
        L += [f"| {s} | {fmt(rep[s][key]['image'])} |" for s in splits]
    if any(rep[s].get("perio") for s in splits):
        L += ["", "## Mức CEJ + tiêu xương nha chu", "",
              f"Model vùng: `{rep.get('perio_weights')}`. GT = `perio_geom` trên vùng bác sĩ (2 vòng CEJ + 1 vòng xương) và CÙNG polygon răng của app "
              "→ sai số chỉ do model vùng. Sai số điểm tính bằng px trên ảnh gốc (~0.1 mm/px trên phim toàn cảnh, tuỳ máy).", "",
              "| Split | Ảnh | IoU thân răng trên | IoU thân răng dưới | IoU vùng xương | Sai số CEJ px (median / p90) | Sai số đỉnh xương px (median / p90) | MAE % tiêu xương | Đúng giai đoạn | Kappa (w) |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for s in splits:
            q = rep[s].get("perio")
            if q:
                i = q["region_iou"]
                L.append(f"| {s} | {q['images']} | {i['crown_upper']:.3f} | {i['crown_lower']:.3f} | {i['bone_window']:.3f} | "
                         f"{q['cej_err_px']['median']} / {q['cej_err_px']['p90']} | {q['crest_err_px']['median']} / {q['crest_err_px']['p90']} | "
                         f"{q['rbl_mae'] * 100:.1f} | {q['stage_acc']:.3f} | {q['stage_kappa']:.3f} |")
        L += ["", "Răng tiêu xương giai đoạn ≥ II (≥ 15%):", "", H]
        L += [f"| {s} | {fmt(rep[s]['perio']['stage2plus'])} |" for s in splits if rep[s].get("perio")]
        if rep.get("test", {}).get("perio"):
            c = rep["test"]["perio"]["stage_confusion"]
            L += ["", "Ma trận giai đoạn (test; hàng = GT, cột = dự đoán):", "", "| GT \\ dự đoán | I | II | III/IV |", "|---|---|---|---|"]
            L += [f"| {n} | {r[0]} | {r[1]} | {r[2]} |" for n, r in zip(("I", "II", "III/IV"), c)]
    elif rep.get("perio_weights") is None:
        L += ["", "## Mức CEJ + tiêu xương nha chu", "", "Chưa có model (research/perio-seg/runs/**/best.pt) → bỏ qua."]
    L += ["", "## Ảnh combine", "", "`vis/<split>/<doc>_<ảnh>.jpg` — trái: nhãn bác sĩ (viền răng xám, vùng sâu đỏ, box răng khôn lệch ngầm đỏ); "
          "phải: dự đoán như webapp (polygon răng màu theo cung hàm + FDI, vùng sâu viền đỏ + conf, ✗ = không khớp nhãn; "
          "trục răng khôn đỏ = ngầm/lệch, xanh lá = bình thường, trục răng 7 xanh dương đứt nét, mặt nhai vàng, CEJ tím). "
          "Nha chu (cả 2 bên): viền tím đứt nét = vùng thân răng, viền cam = vùng xương; chấm tím = CEJ, chấm cam = đỉnh xương; "
          "đoạn nối xanh/cam/đỏ = giai đoạn I/II/III, % hiện cho GĐ ≥ II (∠ = dạng góc)."]
    Path(path).write_text("\n".join(L) + "\n", encoding="utf8")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "final_data"))
    ap.add_argument("--tooth-weights", help="mặc định: best.pt mới nhất trong research/tooth-detect/runs")
    ap.add_argument("--caries-weights", help="mặc định: best.pt mới nhất trong research/caries-seg/runs")
    ap.add_argument("--perio-weights", help="mặc định: best.pt mới nhất trong research/perio-seg/runs (không có -> bỏ qua nha chu)")
    ap.add_argument("--out", default=str(HERE / "out"))
    ap.add_argument("--splits", default="test,val,train", help="split cần chạy")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--wis-iou", type=float, default=0.3)
    ap.add_argument("--car-iou", type=float, default=0.2)
    ap.add_argument("--vis-scale", type=float, default=0.6, help="thu nhỏ ảnh combine")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="chạy lại dự đoán (mặc định dùng cache theo model)")
    a = ap.parse_args()

    tw = Path(a.tooth_weights) if a.tooth_weights else latest_best("research/tooth-detect/runs/**/weights/best.pt")
    cw = Path(a.caries_weights) if a.caries_weights else latest_best("research/caries-seg/runs/**/weights/best.pt")
    pw = Path(a.perio_weights) if a.perio_weights else latest_best("research/perio-seg/runs/**/weights/best.pt", required=False)
    t_onnx, t_run = export_onnx(tw, "tooth")
    c_onnx, c_run = export_onnx(cw, "caries")
    p_onnx, p_run = export_onnx(pw, "perio") if pw else (None, None)
    tag = f"{t_run}__{c_run}" + (f"__{p_run}" if p_run else "")
    print(f"Model răng: {tw}\nModel sâu răng: {cw}\nModel CEJ/xương: {pw or 'chưa có -> bỏ qua'}\n-> {tag}", flush=True)
    # chạy y như app: trỏ đường dẫn model của pipeline vào ONNX vừa export, gọi lần lượt các bước theo STEPS
    pipeline.MODEL_PATH, pipeline.CARIES_MODEL_PATH = str(t_onnx), str(c_onnx)
    pipeline.PERIO_MODEL_PATH = str(p_onnx) if p_onnx else str(HERE / "cache/onnx/_none.onnx")
    cfg = pipeline.resolve_config(None)
    if not p_onnx:
        cfg["cej"]["enabled"] = cfg["perio"]["enabled"] = False

    out = Path(a.out) / tag
    # khoá cache = model + config (đổi mặc định, vd răng khôn chỉ hàm dưới -> chạy lại)
    chash = hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]
    pcache = HERE / "cache/pred" / (f"{t_onnx.stem}__{c_onnx.stem}" + (f"__{p_onnx.stem}" if p_onnx else "") + f"__cfg{chash}")
    pcache.mkdir(parents=True, exist_ok=True)
    want = a.splits.split(",")
    items = [it for it in collect(a.src) if it["split"] in want][: a.limit or None]
    print(f"{len(items)} ảnh ({Counter(it['split'] for it in items)})", flush=True)

    t0, results = time.time(), []
    wrows, crows, lrows, prows = [], [], [], []
    for k, it in enumerate(items):
        img = cv2.imread(it["img"], cv2.IMREAD_COLOR)
        if img is None:
            print("  không đọc được", it["img"]); continue
        H, W = img.shape[:2]
        cp = pcache / f"{it['stem']}.json"
        if cp.exists() and not a.no_cache:
            pred = json.load(open(cp))
        else:
            res = {"width": W, "height": H, "teeth": [], "regions": [], "findings": {}, "config": cfg}
            for key, _ in pipeline.STEPS:
                if cfg[key]["enabled"] and key in pipeline.RUNNERS:
                    pipeline.RUNNERS[key](img, res, cfg[key])
            f = res["findings"]
            pred = dict(teeth=res["teeth"], wisdom=f.get("wisdom", []), caries=f.get("caries", []))
            if p_onnx:
                pred.update(regions=res["regions"], cej=f.get("cej", []), perio=f.get("perio", []))
            json.dump(pred, open(cp, "w"))
        ev = eval_image(it, pred, a)
        ev["perio"] = eval_perio(it, pred, cfg["perio"], W, H)
        # box răng khôn được app phát hiện đúng: box khớp 1 răng app kết luận ngầm/lệch
        ev["wis_box_tp"] = 0
        for j, b in enumerate(it["boxes"]):
            for w in pred["wisdom"]:
                t = next((t for t in pred["teeth"] if t["id"] == w.get("tooth_id")), None)
                if t is None or not w.get("impacted") or w["fdi"] not in ("38", "48"):
                    continue
                poly = np.asarray(t["polygon"]); cx, cy = poly.mean(0)
                if box_iou(bbox(poly), b) >= a.wis_iou or (b[0] <= cx <= b[2] and b[1] <= cy <= b[3]):
                    ev["wis_box_tp"] += 1; break
        ev.update(split=it["split"], n_box=len(it["boxes"]), n_gt_caries=len(it["caries"]), n_pred_caries=len(pred["caries"]))
        results.append(ev)
        base = dict(split=it["split"], doc=it["doc"], image=it["name"])
        wrows += [{**base, **w} for w in ev["wisdom"]]
        crows += [{**base, "caries_labeled": int(len(it["caries"]) > 0), **c} for c in ev["caries_teeth"]]
        lrows += [{**base, **l} for l in ev["lesions"]]
        if ev["perio"]:
            prows += [{**base, **x} for x in ev["perio"]["sides"]]
        if not a.no_vis:
            d = out / "vis" / it["split"]; d.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(d / f"{it['stem']}.jpg"), combine(img, it, pred, ev, a), [cv2.IMWRITE_JPEG_QUALITY, 88])
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(items)} ảnh · {time.time() - t0:.0f}s", flush=True)

    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("wisdom_teeth", wrows), ("caries_teeth", crows), ("caries_lesions", lrows), ("perio_sides", prows)):
        if rows:
            with open(out / f"{name}.csv", "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    groups = {s: [r for r in results if r["split"] == s] for s in want if any(r["split"] == s for r in results)}
    groups["all"] = results
    rep = summarize(groups, dict(tag=tag, date=time.strftime("%Y-%m-%d %H:%M"), n_images=len(results), seconds=time.time() - t0,
                                 tooth_weights=str(tw.relative_to(ROOT)), caries_weights=str(cw.relative_to(ROOT)),
                                 perio_weights=str(pw.relative_to(ROOT)) if pw else None, crop_cfg=load_crop_cfg(str(c_onnx)),
                                 config={k: cfg[k] for k in ("seg", "cej", "wisdom", "caries", "perio")}, wis_iou=a.wis_iou, car_iou=a.car_iou))
    json.dump(rep, open(out / "report.json", "w"), ensure_ascii=False, indent=1)
    write_md(rep, out / "report.md", a)
    print((out / "report.md").read_text())
    print(f"-> {out}")


if __name__ == "__main__":
    main()
