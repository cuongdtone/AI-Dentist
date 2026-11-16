"""Vẽ thử dữ liệu sâu răng theo từng răng (crop) để xem trước khi làm dataset.
Dùng: python preview.py [--n 24] [--mask-outside]

Nguồn:
  DENTEX quadrant-enumeration-disease: polygon nhãn Caries / Deep Caries = VIỀN CẢ RĂNG bị bệnh (không phải vùng sâu).
  final_data 'Sau rang (polygon)': polygon VÙNG SÂU do bác sĩ vẽ; răng chứa nó lấy từ viền 'Truc rang'.
Ra: preview/dentex_sheet.jpg, preview/fd_sheet.jpg, preview/*_full_*.jpg
"""
import argparse, glob, json, os, random, sys
import cv2
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))
DENTEX = os.path.join(ROOT, "../dentex/extracted/training_data/quadrant-enumeration-disease")
FD = os.path.join(ROOT, "../../final_data")
CAT3 = {0: "Impacted", 1: "Caries", 2: "Periapical", 3: "Deep Caries"}
COL = {"Caries": (0, 200, 255), "Deep Caries": (0, 0, 255), "lesion": (0, 0, 255), "tooth": (0, 255, 0)}


def imread(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def crop(img, tooth, margin=0.15, mask_outside=False):
    """Cắt bbox của polygon răng (+margin). mask_outside: tô đen phần ngoài mask răng (mặc định tắt)."""
    H, W = img.shape[:2]
    x, y, w, h = cv2.boundingRect(np.round(tooth).astype(np.int32))
    mx, my = int(w * margin), int(h * margin)
    x1, y1, x2, y2 = max(0, x - mx), max(0, y - my), min(W, x + w + mx), min(H, y + h + my)
    c = img[y1:y2, x1:x2].copy()
    if mask_outside:
        m = np.zeros(c.shape[:2], np.uint8)
        cv2.fillPoly(m, [np.round(tooth - [x1, y1]).astype(np.int32)], 1)
        c[m == 0] = 0
    return c, np.array([x1, y1])


def tile(c, polys, title, size=256):
    s = size / max(c.shape[:2])
    t = cv2.resize(c, (round(c.shape[1] * s), round(c.shape[0] * s)))
    for p, col in polys:
        cv2.polylines(t, [np.round(p * s).astype(np.int32)], True, col, 2)
    out = np.zeros((size + 24, size, 3), np.uint8)
    out[24:24 + t.shape[0], :t.shape[1]] = t
    cv2.putText(out, title, (4, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return out


def sheet(tiles, cols=6):
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    return np.vstack([np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)])


def dentex(n, mask_outside):
    d = json.load(open(f"{DENTEX}/train_quadrant_enumeration_disease.json"))
    files = {im["id"]: im["file_name"] for im in d["images"]}
    anns = [a for a in d["annotations"] if int(a["category_id_3"]) in (1, 3)]
    print(f"DENTEX: {len(anns)} răng Caries/Deep Caries trên {len({a['image_id'] for a in anns})} ảnh")
    random.seed(0)
    tiles, by_img = [], {}
    for a in random.sample(anns, n):
        img = imread(f"{DENTEX}/xrays/{files[int(a['image_id'])]}")
        p = np.array(a["segmentation"][0], float).reshape(-1, 2)
        lab = CAT3[int(a["category_id_3"])]
        fdi = (int(a["category_id_1"]) + 1) * 10 + int(a["category_id_2"]) + 1
        c, o = crop(img, p, mask_outside=mask_outside)
        tiles.append(tile(c, [(p - o, COL[lab])], f"{fdi} {lab}"))
        by_img.setdefault(int(a["image_id"]), []).append(a)
    cv2.imwrite(f"{ROOT}/preview/dentex_sheet.jpg", sheet(tiles))
    # 1 ảnh đầy đủ: mọi nhãn bệnh của ảnh đó
    iid = max(by_img, key=lambda k: sum(int(a["image_id"]) == k for a in d["annotations"]))
    img = imread(f"{DENTEX}/xrays/{files[iid]}")
    for a in d["annotations"]:
        if int(a["image_id"]) == iid:
            lab = CAT3[int(a["category_id_3"])]
            p = np.array(a["segmentation"][0], np.int32).reshape(-1, 2)
            cv2.polylines(img, [p], True, COL.get(lab, (255, 0, 255)), 3)
            cv2.putText(img, lab, tuple(p.min(0)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite(f"{ROOT}/preview/dentex_full_{files[iid]}.jpg", img)


def final_data(n, mask_outside):
    items, n_les, n_free = [], 0, 0
    for af in sorted(glob.glob(f"{FD}/*/ann/*.json")):
        objs = json.load(open(af, encoding="utf8"))["objects"]
        les = [np.array(o["points"]["exterior"], float) for o in objs if o["classTitle"].startswith("Sau rang")]
        teeth = [np.array(o["points"]["exterior"], float) for o in objs
                 if o["classTitle"] == "Truc rang" and o["geometryType"] == "line" and len(o["points"]["exterior"]) >= 3]
        n_les += len(les)
        for t in teeth:
            inside = [l for l in les if cv2.pointPolygonTest(t.astype(np.float32), tuple(map(float, l.mean(0))), False) >= 0]
            if inside:
                items.append((af, t, inside))
        n_free += sum(all(cv2.pointPolygonTest(t.astype(np.float32), tuple(map(float, l.mean(0))), False) < 0 for t in teeth) for l in les)
    print(f"final_data: {n_les} vùng sâu, {len(items)} răng có sâu (có viền Truc rang), "
          f"{n_free} vùng sâu không nằm trong viền răng nào")
    random.seed(0)
    tiles = []
    for af, t, les in random.sample(items, n):
        img = imread(os.path.join(os.path.dirname(os.path.dirname(af)), "img", os.path.basename(af)[:-5]))
        c, o = crop(img, t, mask_outside=mask_outside)
        doc = af.split(os.sep)[-3]
        tiles.append(tile(c, [(t - o, COL["tooth"])] + [(l - o, COL["lesion"]) for l in les], f"{doc} {len(les)} lesion"))
    cv2.imwrite(f"{ROOT}/preview/fd_sheet.jpg", sheet(tiles))
    af = items[0][0]
    img = imread(os.path.join(os.path.dirname(os.path.dirname(af)), "img", os.path.basename(af)[:-5]))
    for o in json.load(open(af, encoding="utf8"))["objects"]:
        p = np.array(o["points"]["exterior"], np.int32)
        if o["classTitle"].startswith("Sau rang"):
            cv2.polylines(img, [p], True, COL["lesion"], 3)
        elif o["classTitle"] == "Truc rang" and len(p) >= 3:
            cv2.polylines(img, [p], True, COL["tooth"], 1)
    cv2.imwrite(f"{ROOT}/preview/fd_full_{af.split(os.sep)[-3]}.jpg", img)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--mask-outside", action="store_true", help="che phần ngoài mask răng (mặc định tắt)")
    a = ap.parse_args()
    os.makedirs(f"{ROOT}/preview", exist_ok=True)
    dentex(a.n, a.mask_outside)
    final_data(a.n, a.mask_outside)
