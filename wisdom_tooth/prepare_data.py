"""Chuyển final_data (Supervisely) -> dataset YOLO 1 lớp: răng khôn hàm dưới lệch/ngầm.
Dùng: python prepare_data.py --src ../final_data --out data --max-side 1600
- Gộp 3 thư mục bác sĩ (Lan, Minh, Thanh), chia train/val/test phân tầng theo số răng khôn/ảnh.
- Ảnh trùng hash luôn nằm cùng 1 tập (tránh rò rỉ dữ liệu).
- Ảnh không có răng khôn được giữ lại làm mẫu âm (file nhãn rỗng).
"""
import argparse, json, os, random, shutil, sys
from collections import defaultdict
import cv2
import numpy as np

CLASS = "Rang khon ham duoi lech ngam"
sys.stdout.reconfigure(encoding="utf-8")


def collect(src):
    items = []
    for doc in sorted(os.listdir(src)):
        d = os.path.join(src, doc)
        if not os.path.isdir(os.path.join(d, "ann")):
            continue
        for f in sorted(os.listdir(os.path.join(d, "ann"))):
            name = f[:-5]
            ann = json.load(open(os.path.join(d, "ann", f), encoding="utf8"))
            info_p = os.path.join(d, "img_info", f)
            h = json.load(open(info_p, encoding="utf8"))["hash"] if os.path.exists(info_p) else f"{doc}/{name}"
            W, H = ann["size"]["width"], ann["size"]["height"]
            boxes = []
            for o in ann["objects"]:
                if o["classTitle"] != CLASS:
                    continue
                pts = np.array(o["points"]["exterior"], float)
                x1, y1 = pts.min(0); x2, y2 = pts.max(0)
                x1, x2 = np.clip([x1, x2], 0, W); y1, y2 = np.clip([y1, y2], 0, H)
                if x2 - x1 > 2 and y2 - y1 > 2:
                    boxes.append(((x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H))
            items.append(dict(doc=doc, name=name, img=os.path.join(d, "img", name), hash=h, boxes=boxes))
    return items


def split(items, ratios, seed):
    groups = defaultdict(list)
    for it in items:
        groups[it["hash"]].append(it)
    strata = defaultdict(list)  # phân tầng theo số răng khôn (0/1/2) của nhóm
    for g in groups.values():
        strata[min(len(g[0]["boxes"]), 2)].append(g)
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for k in sorted(strata):
        gs = strata[k]; rng.shuffle(gs)
        n_val, n_test = round(len(gs) * ratios[1]), round(len(gs) * ratios[2])
        parts = {"val": gs[:n_val], "test": gs[n_val:n_val + n_test], "train": gs[n_val + n_test:]}
        for s, p in parts.items():
            out[s] += [it for g in p for it in g]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../final_data")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-side", type=int, default=1600, help="thu nhỏ ảnh để train nhanh hơn; 0 = giữ nguyên")
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    items = collect(a.src)
    sets = split(items, (1 - a.val - a.test, a.val, a.test), a.seed)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    for s, its in sets.items():
        os.makedirs(f"{a.out}/images/{s}"); os.makedirs(f"{a.out}/labels/{s}")
        for it in its:
            stem = f"{it['doc']}_{os.path.splitext(it['name'])[0]}"
            img = cv2.imdecode(np.fromfile(it["img"], np.uint8), cv2.IMREAD_COLOR)
            if a.max_side and max(img.shape[:2]) > a.max_side:
                r = a.max_side / max(img.shape[:2])
                img = cv2.resize(img, None, fx=r, fy=r, interpolation=cv2.INTER_AREA)
            cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tofile(f"{a.out}/images/{s}/{stem}.jpg")
            with open(f"{a.out}/labels/{s}/{stem}.txt", "w") as f:
                f.writelines(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in it["boxes"])
        n_pos = sum(bool(i["boxes"]) for i in its)
        print(f"{s:5s}: {len(its)} ảnh ({n_pos} có răng khôn, {sum(len(i['boxes']) for i in its)} box)")

    root = os.path.abspath(a.out).replace("\\", "/")
    with open(f"{a.out}/data.yaml", "w", encoding="utf8") as f:
        f.write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: rang_khon_ham_duoi\n")
    print("đã ghi", f"{a.out}/data.yaml")


if __name__ == "__main__":
    main()
