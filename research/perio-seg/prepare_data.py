"""Tạo dataset YOLO-seg mức CEJ + mức xương nha chu từ final_data (ảnh panoramic nguyên khung).

Nhãn bác sĩ là VÙNG khép kín theo cả cung hàm (không theo từng răng):
  - 'Muc CEJ'          : 2 vòng / ảnh = vùng thân răng hàm trên / hàm dưới (mép phía chóp = đường CEJ).
  - 'Muc xuong nha chu': 1 vòng / ảnh = vùng giữa đỉnh xương ổ răng hàm trên và hàm dưới
                         (mép trên / dưới = đường đỉnh xương).
-> 3 lớp: 0 crown_upper, 1 crown_lower, 2 bone_window (vòng CEJ trên/dưới phân theo y trung bình).
Ảnh không có đúng 2 vòng CEJ + 1 vòng xương bị bỏ (ảnh trẻ em / chưa gán / vẽ lỗi). Ảnh trùng hash chỉ lấy 1 lần.
Split theo wisdom_tooth/data (test không train). Ảnh được symlink (không chép).

Dùng: python prepare_data.py [--out data] [--vis 30]
Ra: <out>/{images,labels}/{train,val,test}, data.yaml, meta.csv, report.json, vis/
"""
import argparse, csv, json, os, shutil, sys
from collections import Counter
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
NAMES = ["crown_upper", "crown_lower", "bone_window"]
CEJ, BONE = "Muc CEJ", "Muc xuong nha chu"
COLORS = [(239, 70, 217), (246, 92, 139), (22, 115, 249)]  # BGR: tím, tím hồng, cam


def split_names(wdir):
    return {os.path.splitext(f)[0]: s for s in ("train", "val", "test")
            for f in os.listdir(os.path.join(wdir, "images", s))}


def parse(ann):
    """-> ({lớp: polygon}, lý do bỏ hoặc None)"""
    cej = [np.array(o["points"]["exterior"], float) for o in ann["objects"] if o["classTitle"] == CEJ and len(o["points"]["exterior"]) >= 3]
    bone = [np.array(o["points"]["exterior"], float) for o in ann["objects"] if o["classTitle"] == BONE and len(o["points"]["exterior"]) >= 3]
    if not cej and not bone:
        return None, "không có nhãn CEJ / xương"
    if len(cej) != 2:
        return None, f"{len(cej)} vòng CEJ"
    if len(bone) != 1:
        return None, f"{len(bone)} vòng xương"
    up, lo = sorted(cej, key=lambda p: p[:, 1].mean())
    if up[:, 1].mean() >= lo[:, 1].mean():
        return None, "vòng CEJ trên/dưới trùng nhau"
    return {0: up, 1: lo, 2: bone[0]}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "../../final_data"))
    ap.add_argument("--split-from", default=os.path.join(ROOT, "../../wisdom_tooth/data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--vis", type=int, default=30, help="số ảnh vẽ kiểm tra (train)")
    a = ap.parse_args()

    split = split_names(a.split_from)
    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    for s in ("train", "val", "test"):
        os.makedirs(f"{a.out}/images/{s}"); os.makedirs(f"{a.out}/labels/{s}")
    os.makedirs(f"{a.out}/vis")

    seen, st, kept, rows, nvis = set(), Counter(), Counter(), [], 0
    for doc in sorted(os.listdir(a.src)):
        d = os.path.join(a.src, doc)
        if not os.path.isdir(os.path.join(d, "ann")):
            continue
        for f in sorted(os.listdir(os.path.join(d, "ann"))):
            name = f[:-5]
            info = os.path.join(d, "img_info", f)
            h = json.load(open(info, encoding="utf8")).get("hash") if os.path.exists(info) else f"{doc}/{name}"
            if h in seen:
                st["trùng hash"] += 1; continue
            seen.add(h)
            stem = f"{doc}_{os.path.splitext(name)[0]}"
            s = split.get(stem)
            if s is None:
                st["không có trong split"] += 1; continue
            ann = json.load(open(os.path.join(d, "ann", f), encoding="utf8"))
            polys, why = parse(ann)
            if why:
                st[why] += 1; continue
            W, H = ann["size"]["width"], ann["size"]["height"]
            img = os.path.abspath(os.path.join(d, "img", name))
            ext = os.path.splitext(name)[1]
            os.symlink(img, f"{a.out}/images/{s}/{stem}{ext}")
            with open(f"{a.out}/labels/{s}/{stem}.txt", "w") as fh:
                for c, p in polys.items():
                    q = np.clip(p / [W, H], 0, 1)
                    fh.write(f"{c} " + " ".join(f"{v:.5f}" for v in q.reshape(-1)) + "\n")
            kept[s] += 1
            rows.append([s, stem, W, H] + [len(polys[c]) for c in range(3)])
            if nvis < a.vis and s == "train":
                im = cv2.imread(img)
                for c, p in polys.items():
                    cv2.polylines(im, [np.round(p).astype(np.int32)], True, COLORS[c], 3, cv2.LINE_AA)
                cv2.imwrite(f"{a.out}/vis/{stem}.jpg", cv2.resize(im, None, fx=0.5, fy=0.5)); nvis += 1

    with open(f"{a.out}/meta.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["split", "stem", "w", "h", "pts_crown_upper", "pts_crown_lower", "pts_bone_window"]); w.writerows(rows)
    with open(f"{a.out}/data.yaml", "w") as fh:
        fh.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
                 + "".join(f"  {i}: {n}\n" for i, n in enumerate(NAMES)))
    rep = {"kept": dict(kept), "skipped": dict(st), "names": NAMES}
    json.dump(rep, open(f"{a.out}/report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
