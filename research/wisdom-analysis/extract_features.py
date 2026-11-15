"""Chạy model FDI-seg lên ảnh DENTEX có nhãn bệnh -> đặc trưng hình học + nhãn ngầm cho từng răng khôn.
Dùng:
  python extract_features.py --split train --weights <best.pt>   # 705 ảnh quadrant-enumeration-disease
  python extract_features.py --split test  --weights <best.pt>   # 250 ảnh test (LabelMe)

LƯU Ý: các thư mục con của DENTEX là các bộ ảnh KHÁC NHAU dù trùng tên file -> không ghép nhãn
theo tên. Ở đây nhãn lấy từ chính ảnh đó: răng khôn dự đoán được coi là "ngầm" nếu mask của nó
trùng (IoU >= --iou) với một polygon 'Impacted' (train) / 'gömülü' (test); ngược lại là "không ngầm"
(giả định bác sĩ DENTEX đã gán đủ mọi răng ngầm trong ảnh).
"""
import argparse, csv, glob, json, os, sys
import cv2
import numpy as np
from ultralytics import YOLO
from wisdom_geom import DEFAULT_PARAMS, WISDOM, analyze_tooth, draw, predict_masks

sys.stdout.reconfigure(encoding="utf-8")


def iter_train(src):
    D = json.load(open(f"{src}/training_data/quadrant-enumeration-disease/train_quadrant_enumeration_disease.json"))
    c1 = {c["id"]: int(c["name"]) for c in D["categories_1"]}
    c2 = {c["id"]: int(c["name"]) for c in D["categories_2"]}
    c3 = {c["id"]: c["name"] for c in D["categories_3"]}
    by_img = {i["id"]: [] for i in D["images"]}
    for a in D["annotations"]:
        if c3[a["category_id_3"]] == "Impacted":
            by_img[a["image_id"]].append((c1[a["category_id_1"]] * 10 + c2[a["category_id_2"]],
                                         np.array(a["segmentation"][0], float).reshape(-1, 2)))
    for im in D["images"]:
        yield f"{src}/training_data/quadrant-enumeration-disease/xrays/{im['file_name']}", by_img[im["id"]]


def iter_test(src):
    for jf in sorted(glob.glob(f"{src}/disease/label/*.json")):
        lab = json.load(open(jf, encoding="utf8"))
        imp = [(int(s["label"].split("-")[2]), np.array(s["points"], float))
               for s in lab["shapes"] if s["label"].split("-")[1] == "gömülü"]
        yield f"{src}/disease/input/{os.path.basename(jf)[:-5]}.png", imp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "test"], required=True)
    ap.add_argument("--weights", default="../tooth-detect/runs/segment/runs/yolo26s-seg_1024/weights/best.pt")
    ap.add_argument("--src", default="../dentex/extracted")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3)
    ap.add_argument("--device", default=None)
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu (kiểm thử)")
    ap.add_argument("--vis", default=None, help="thư mục lưu ảnh vẽ kết quả (tuỳ chọn)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or f"features_{a.split}.csv"
    model = YOLO(a.weights)
    if a.vis:
        os.makedirs(a.vis, exist_ok=True)

    rows, n_gt, n_gt_hit = [], 0, 0
    it = iter_train(a.src) if a.split == "train" else iter_test(a.src)
    for k, (img_p, imp) in enumerate(it):
        if a.limit and k >= a.limit:
            break
        img = cv2.imread(img_p); H, W = img.shape[:2]
        masks = predict_masks(model, img, a.imgsz, a.conf, a.device)
        gts = []
        for f, poly in imp:
            m = np.zeros((H, W), np.uint8); cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
            gts.append((f, m.astype(bool)))
        n_gt += len(gts)
        hit = set()
        res = []
        for f in WISDOM:
            if f not in masks:
                continue
            r = analyze_tooth(masks[f], masks.get(f - 1), f, DEFAULT_PARAMS)
            res.append(r)
            if "winter" not in r:
                continue
            ious = [((g & masks[f]).sum() / max((g | masks[f]).sum(), 1), j) for j, (_, g) in enumerate(gts)]
            best = max(ious, default=(0, -1))
            lab = int(best[0] >= a.iou)
            if lab:
                hit.add(best[1])
            rows.append(dict(image=os.path.basename(img_p), fdi=f, label=lab,
                             gt_fdi=gts[best[1]][0] if lab else "", angle=r["winter_angle"],
                             depth_ratio=r.get("depth_ratio", ""), winter=r["winter"],
                             m3_elong=r["m3_elong"], flags="|".join(r["flags"])))
        n_gt_hit += len(hit)
        if a.vis:
            cv2.imwrite(os.path.join(a.vis, os.path.basename(img_p)[:-4] + ".jpg"), draw(img, res, masks))
        if (k + 1) % 50 == 0:
            print(f"  {k + 1} ảnh...", flush=True)

    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    y = sum(r["label"] for r in rows)
    print(f"[{a.split}] răng khôn dự đoán: {len(rows)} | ghép được nhãn ngầm: {y}")
    print(f"  răng ngầm GT: {n_gt}, được model tìm thấy (IoU>={a.iou}): {n_gt_hit} ({n_gt_hit / max(n_gt, 1):.1%})")
    print(f"  -> {out}")


if __name__ == "__main__":
    main()
