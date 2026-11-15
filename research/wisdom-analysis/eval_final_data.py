"""Đánh giá luật ngầm/lệch trên final_data (ảnh + nhãn bác sĩ bên mình, định dạng Supervisely).
Dùng: python eval_final_data.py --src ../../final_data --params params.json [--vis out_final]

Nhãn: box 'Rang khon ham duoi lech ngam' (chỉ hàm dưới). Với mỗi răng 38/48 model tìm thấy:
  ngầm/lệch (1) nếu bbox của mask khớp box bác sĩ (IoU >= --iou hoặc tâm mask nằm trong box), ngược lại 0.
Giả định bác sĩ khoanh đủ mọi răng khôn hàm dưới lệch ngầm. Ảnh trùng hash chỉ tính 1 lần.
"""
import argparse, csv, json, os, sys
from collections import Counter
import cv2
import numpy as np
from ultralytics import YOLO
from wisdom_geom import analyze_tooth, draw, load_params, predict_masks

sys.stdout.reconfigure(encoding="utf-8")
CLASS = "Rang khon ham duoi lech ngam"


def collect(src):
    seen, items = set(), []
    for doc in sorted(os.listdir(src)):
        d = os.path.join(src, doc)
        if not os.path.isdir(os.path.join(d, "ann")):
            continue
        for f in sorted(os.listdir(os.path.join(d, "ann"))):
            name = f[:-5]
            info = os.path.join(d, "img_info", f)
            h = json.load(open(info, encoding="utf8")).get("hash") if os.path.exists(info) else f"{doc}/{name}"
            if h in seen:
                continue
            seen.add(h)
            ann = json.load(open(os.path.join(d, "ann", f), encoding="utf8"))
            boxes = []
            for o in ann["objects"]:
                if o["classTitle"] == CLASS:
                    (x1, y1), (x2, y2) = o["points"]["exterior"][:2]
                    boxes.append((min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)))
            items.append(dict(doc=doc, name=name, img=os.path.join(d, "img", name), boxes=boxes))
    return items


def box_iou(a, b):
    iw = max(0, min(a[2], b[2]) - max(a[0], b[0])); ih = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = iw * ih
    return i / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i, 1)


def prf(y, yh):
    y, yh = np.array(y), np.array(yh)
    tp = int(((y == 1) & (yh == 1)).sum()); fp = int(((y == 0) & (yh == 1)).sum()); fn = int(((y == 1) & (yh == 0)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return f"P={P:.3f} R={R:.3f} F1={2 * P * R / max(P + R, 1e-9):.3f} Acc={(y == yh).mean():.3f} (n={len(y)}, TP={tp} FP={fp} FN={fn})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../../final_data")
    ap.add_argument("--weights", default="../tooth-detect/runs/segment/runs/yolo26s-seg_1024/weights/best.pt")
    ap.add_argument("--params", default="params.json")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vis", default=None, help="lưu ảnh vẽ các ca sai (FP/FN)")
    ap.add_argument("--out", default="features_final_data.csv")
    a = ap.parse_args()

    p = load_params(a.params); model = YOLO(a.weights)
    items = collect(a.src)[: a.limit or None]
    if a.vis:
        os.makedirs(a.vis, exist_ok=True)
    rows, n_box, n_box_hit = [], 0, 0
    for k, it in enumerate(items):
        img = cv2.imread(it["img"])
        if img is None:
            continue
        masks = predict_masks(model, img, a.imgsz, 0.25, a.device)
        n_box += len(it["boxes"]); hit = set(); res_img = []
        for f in (38, 48):
            if f not in masks:
                continue
            r = analyze_tooth(masks[f], masks.get(f - 1), f, p)
            if "winter" not in r:
                continue
            ys, xs = np.nonzero(masks[f]); mb = (xs.min(), ys.min(), xs.max(), ys.max()); cx, cy = xs.mean(), ys.mean()
            lab, j = 0, -1
            for i, b in enumerate(it["boxes"]):
                if box_iou(mb, b) >= a.iou or (b[0] <= cx <= b[2] and b[1] <= cy <= b[3]):
                    lab, j = 1, i; break
            if lab:
                hit.add(j)
            r["label"] = lab; res_img.append(r)
            rows.append(dict(doc=it["doc"], image=it["name"], fdi=f, label=lab, pred=int(r["impacted"]),
                             angle=r["winter_angle"], depth_ratio=r.get("depth_ratio", ""), winter=r["winter"],
                             depth=r["depth"] or "", flags="|".join(r["flags"])))
        n_box_hit += len(hit)
        if a.vis and any(r["label"] != r["impacted"] for r in res_img):
            vis = draw(img, res_img, masks)
            for b in it["boxes"]:
                cv2.rectangle(vis, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 165, 255), 3)
            cv2.imwrite(os.path.join(a.vis, f"{it['doc']}_{os.path.splitext(it['name'])[0]}.jpg"), vis)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(items)} ảnh...", flush=True)

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    y = [r["label"] for r in rows]; yh = [r["pred"] for r in rows]
    print(f"Ảnh (sau khi bỏ trùng): {len(items)} | box bác sĩ: {n_box}, model tìm thấy răng 38/48 khớp: {n_box_hit} ({n_box_hit / max(n_box, 1):.1%})")
    print(f"Răng 38/48 model tìm thấy: {len(rows)}, bác sĩ gán lệch ngầm: {sum(y)}")
    print("Luật hiện tại (params.json):", prf(y, yh))
    print("Winter:", Counter(r["winter"] for r in rows).most_common(), "| độ sâu:", Counter(r["depth"] for r in rows).most_common())
    for f in (38, 48):
        sub = [r for r in rows if r["fdi"] == f]
        print(f"  {f}:", prf([r["label"] for r in sub], [r["pred"] for r in sub]))
    # ngưỡng tối ưu NGAY TRÊN final_data (chỉ để tham khảo mức trần, KHÔNG phải kết quả đánh giá)
    ang = np.abs([r["angle"] for r in rows]); dep = np.array([float(r["depth_ratio"]) if r["depth_ratio"] != "" else -9 for r in rows])
    best = max(((float(prf_f1(y, ((dep > d) | (ang > t)).astype(int))), d, t) for d in np.arange(-0.2, 0.61, 0.01) for t in range(10, 181, 2)))
    print(f"Tham khảo - ngưỡng tốt nhất nếu chỉnh trên chính final_data: F1={best[0]:.3f} (depth>{best[1]:.2f}, góc>{best[2]}°)")


def prf_f1(y, yh):
    y, yh = np.array(y), np.array(yh)
    tp = ((y == 1) & (yh == 1)).sum(); fp = ((y == 0) & (yh == 1)).sum(); fn = ((y == 1) & (yh == 0)).sum()
    return 2 * tp / max(2 * tp + fp + fn, 1)


if __name__ == "__main__":
    main()
