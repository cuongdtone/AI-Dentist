"""DENTEX quadrant_enumeration (COCO) -> dataset YOLO-seg 32 lớp FDI (11..48).
Dùng: python prepare_data.py --src ../dentex/extracted --out data --max-side 1600

- Lớp: idx = (quadrant-1)*8 + (số răng-1)  ->  tên FDI = quadrant*10 + số răng.
- Chia train/val theo ảnh (mặc định 10% val, seed cố định).
- Lật ngang làm OFFLINE với đổi lớp trái<->phải (Q1<->Q2, Q3<->Q4), vì fliplr của
  Ultralytics không đổi lớp -> khi train phải để fliplr=0.
- Test: 250 ảnh DENTEX test (LabelMe) chỉ gán nhãn RĂNG BỆNH kèm số FDI -> không dùng
  để tính mAP (thiếu răng lành), mà để eval_test.py đo độ chính xác đánh số.
"""
import argparse, json, os, random, shutil, sys
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
NAMES = [str(q * 10 + t) for q in range(1, 5) for t in range(1, 9)]
FLIP_Q = {1: 2, 2: 1, 3: 4, 4: 3}


def cls_idx(q, t):
    return (q - 1) * 8 + (t - 1)


def flip_cls(c):
    q, t = c // 8 + 1, c % 8 + 1
    return cls_idx(FLIP_Q[q], t)


def load_coco(src):
    d = json.load(open(os.path.join(src, "training_data/quadrant_enumeration/train_quadrant_enumeration.json")))
    # categories_1: id 0..3 -> tên "1".."4"; categories_2: id 0..7 -> "1".."8" (map theo TÊN cho chắc)
    q_of = {c["id"]: int(c["name"]) for c in d["categories_1"]}
    t_of = {c["id"]: int(c["name"]) for c in d["categories_2"]}
    imgs = {i["id"]: dict(i, objs=[]) for i in d["images"]}
    for a in d["annotations"]:
        im = imgs[a["image_id"]]
        poly = np.array(a["segmentation"][0], float).reshape(-1, 2)
        if len(poly) < 3:
            continue
        im["objs"].append((cls_idx(q_of[a["category_id_1"]], t_of[a["category_id_2"]]), poly))
    return list(imgs.values())


def write_sample(img, objs, W, H, out_img, out_lbl, scale):
    if scale < 1:
        img = cv2.resize(img, (round(W * scale), round(H * scale)), interpolation=cv2.INTER_AREA)
    cv2.imwrite(out_img, img)
    with open(out_lbl, "w") as f:
        for c, poly in objs:
            p = poly / [W, H]
            p = np.clip(p, 0, 1)
            f.write(f"{c} " + " ".join(f"{v:.6f}" for v in p.reshape(-1)) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="../dentex/extracted")
    ap.add_argument("--out", default="data")
    ap.add_argument("--max-side", type=int, default=1600, help="thu nhỏ ảnh để train nhanh (0 = giữ nguyên)")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-flip", action="store_true", help="không tạo bản lật ngang cho train")
    a = ap.parse_args()

    items = sorted(load_coco(a.src), key=lambda x: x["file_name"])
    random.Random(a.seed).shuffle(items)
    n_val = round(len(items) * a.val)
    splits = {"val": items[:n_val], "train": items[n_val:]}

    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    xdir = os.path.join(a.src, "training_data/quadrant_enumeration/xrays")
    n_obj = {"train": 0, "val": 0}
    for s, its in splits.items():
        os.makedirs(f"{a.out}/images/{s}"); os.makedirs(f"{a.out}/labels/{s}")
        for it in its:
            img = cv2.imread(os.path.join(xdir, it["file_name"]))
            H, W = img.shape[:2]
            assert (W, H) == (it["width"], it["height"]), it["file_name"]
            scale = min(1.0, a.max_side / max(W, H)) if a.max_side else 1.0
            stem = os.path.splitext(it["file_name"])[0]
            write_sample(img, it["objs"], W, H, f"{a.out}/images/{s}/{stem}.png", f"{a.out}/labels/{s}/{stem}.txt", scale)
            n_obj[s] += len(it["objs"])
            if s == "train" and not a.no_flip:
                fobjs = [(flip_cls(c), np.stack([W - p[:, 0], p[:, 1]], 1)) for c, p in it["objs"]]
                write_sample(cv2.flip(img, 1), fobjs, W, H, f"{a.out}/images/{s}/{stem}_flip.png",
                             f"{a.out}/labels/{s}/{stem}_flip.txt", scale)
                n_obj[s] += len(fobjs)

    with open(f"{a.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\n")
        f.write(f"nc: {len(NAMES)}\nnames:\n" + "".join(f"  {i}: '{n}'\n" for i, n in enumerate(NAMES)))
    with open(f"{a.out}/split.json", "w") as f:
        json.dump({s: [it["file_name"] for it in its] for s, its in splits.items()}, f, indent=1)
    print(f"train: {len(splits['train'])} ảnh (+flip={not a.no_flip}), {n_obj['train']} răng | "
          f"val: {len(splits['val'])} ảnh, {n_obj['val']} răng -> {a.out}/data.yaml")


if __name__ == "__main__":
    main()
