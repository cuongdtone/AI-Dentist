"""Tạo dataset YOLO-seg vùng sâu răng (1 lớp 'caries') trên ảnh CẮT TỪNG RĂNG từ final_data.
Dùng: python prepare_data.py [--mask-outside] [--neg-ratio 1.0] [--vis 40]

Mỗi ảnh final_data:
  - Răng = viền 'Truc rang' của bác sĩ; răng bác sĩ không vẽ viền -> lấy mask model FDI (conf >= --tooth-conf).
  - Vùng sâu = polygon 'Sau rang' của bác sĩ.
  - Mỗi răng cắt theo bbox + margin. Nhãn = MỌI vùng sâu lọt trong khung cắt (kể cả của răng kề,
    để không dạy model rằng vùng sâu thấy được là nền). --mask-outside: tô đen ngoài mask răng và
    chỉ giữ phần vùng sâu nằm trong răng (mặc định tắt).
  - Răng có vùng sâu = mẫu dương; răng không có = mẫu âm (ảnh không có file nhãn), lấy ngẫu nhiên
    tối đa --neg-ratio x số mẫu dương mỗi split.
  - Mẫu âm chỉ lấy từ ảnh mà bác sĩ CÓ vẽ 'Sau rang' (ảnh không vẽ sâu răng có thể là chưa khám sâu).
Split train/val/test theo wisdom_tooth/data (cùng ảnh gốc -> cùng split, không rò rỉ).
Ra: data/{images,labels}/{train,val,test}, data/data.yaml, data/meta.csv, data/report.json, data/vis/
"""
import argparse, csv, glob, json, os, random, shutil, sys
from collections import Counter
import cv2
import numpy as np
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8")
ROOT = os.path.dirname(os.path.abspath(__file__))


def imread(p):
    return cv2.imdecode(np.fromfile(p, np.uint8), cv2.IMREAD_COLOR)


def split_names(wdir):
    return {os.path.splitext(f)[0]: s for s in ("train", "val", "test")
            for f in os.listdir(os.path.join(wdir, "images", s))}


def fill(poly, H, W):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)


def model_teeth(model, img, imgsz, conf, ios_thr=0.6):
    """Mask răng của model FDI, bỏ phát hiện trùng (như app/backend model._dedupe)."""
    r = model.predict(img, imgsz=imgsz, conf=0.25, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []
    M = r.masks.data.cpu().numpy() > 0.5
    S = r.boxes.conf.cpu().numpy()
    F = [model.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
    keep = []
    for i in np.argsort(-S):
        if S[i] < conf or M[i].sum() == 0:
            continue
        if all((M[i] & M[j]).sum() / min(M[i].sum(), M[j].sum()) < ios_thr for j in keep):
            keep.append(i)
    return [(M[i], F[i], float(S[i])) for i in keep]


def mask_polys(m, min_area=8):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cs:
        if cv2.contourArea(c) < min_area:
            continue
        c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2)
        if len(c) >= 3:
            out.append(c.astype(float))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "../../final_data"))
    ap.add_argument("--split-from", default=os.path.join(ROOT, "../../wisdom_tooth/data"))
    ap.add_argument("--tooth-weights", default=os.path.join(ROOT, "../tooth-detect/runs/segment/runs/ft_finaldata_30/weights/best.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--tooth-conf", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.15)
    ap.add_argument("--mask-outside", action="store_true", help="che phần ngoài mask răng (mặc định tắt)")
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="số mẫu âm / mẫu dương mỗi split")
    ap.add_argument("--min-lesion-frac", type=float, default=0.3,
                    help="vùng sâu bị khung cắt cắt mất > (1-x) diện tích thì bỏ phần đó")
    ap.add_argument("--vis", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if os.path.exists(a.out):
        shutil.rmtree(a.out)
    for s in ("train", "val", "test"):
        for d in ("images", "labels"):
            os.makedirs(f"{a.out}/{d}/{s}")
    os.makedirs(f"{a.out}/vis")

    split = split_names(a.split_from)
    model = YOLO(a.tooth_weights)
    st, pos, neg = Counter(), [], []
    anns = sorted(glob.glob(f"{a.src}/*/ann/*.json"))[: a.limit or None]
    for k, af in enumerate(anns):
        doc, name = af.split(os.sep)[-3], os.path.basename(af)[:-5]
        stem = f"{doc}_{os.path.splitext(name)[0]}"
        s = split.get(stem)
        if s is None:
            st["ảnh không có trong split"] += 1; continue
        objs = json.load(open(af, encoding="utf8"))["objects"]
        les = [np.array(o["points"]["exterior"], float) for o in objs
               if o["classTitle"].startswith("Sau rang") and len(o["points"]["exterior"]) >= 3]
        if not les:
            st["ảnh không vẽ sâu răng (bỏ)"] += 1; continue
        img = imread(os.path.join(os.path.dirname(os.path.dirname(af)), "img", name))
        H, W = img.shape[:2]
        teeth = [(fill(p, H, W), "", "doc") for o in objs
                 if o["classTitle"] == "Truc rang" and o["geometryType"] == "line"
                 and len(p := np.array(o["points"]["exterior"], float)) >= 3]
        teeth = [t for t in teeth if t[0].sum() > 0]
        for m, fdi, _ in model_teeth(model, img, a.imgsz, a.tooth_conf):
            if all((m & d).sum() / m.sum() < 0.3 for d, _, _ in teeth):
                teeth.append((m, fdi, "model"))
        L = [fill(p, H, W) for p in les]
        lmask = np.any(L, 0)
        covered = np.zeros(len(L), bool)
        st["vùng sâu"] += len(L)
        for ti, (tm, fdi, src) in enumerate(teeth):
            ys, xs = np.nonzero(tm)
            x1, y1, x2, y2 = xs.min(), ys.min(), xs.max() + 1, ys.max() + 1
            mx, my = int((x2 - x1) * a.margin), int((y2 - y1) * a.margin)
            x1, y1, x2, y2 = max(0, x1 - mx), max(0, y1 - my), min(W, x2 + mx), min(H, y2 + my)
            own = [i for i, l in enumerate(L) if (l & tm).sum() / l.sum() >= 0.5]  # vùng sâu thuộc răng này
            covered[own] = True
            # vùng sâu trong khung: giữ nếu phần lọt khung đủ lớn
            keep = np.zeros((y2 - y1, x2 - x1), bool)
            for l in L:
                part = l[y1:y2, x1:x2]
                if a.mask_outside:
                    part = part & tm[y1:y2, x1:x2]
                if part.sum() >= a.min_lesion_frac * l.sum() and part.sum() > 0:
                    keep |= part
            item = dict(split=s, stem=stem, tooth=ti, src=src, fdi=fdi, box=(x1, y1, x2, y2),
                        own=len(own), polys=mask_polys(keep), af=af, img_name=name,
                        tmask=tm[y1:y2, x1:x2].copy() if a.mask_outside else None)
            (pos if own else neg).append(item)
            # răng không có vùng sâu của chính nó nhưng khung chứa vùng sâu răng kề -> vẫn là mẫu dương (có nhãn)
            if not own and item["polys"]:
                neg.pop(); pos.append(item); st["răng lành nhưng khung có sâu răng kề"] += 1
        st["vùng sâu không thuộc răng nào"] += int((~covered).sum())
        st["răng nguồn " + "doc"] += sum(t[2] == "doc" for t in teeth)
        st["răng nguồn " + "model"] += sum(t[2] == "model" for t in teeth)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(anns)}...", flush=True)

    random.seed(0)
    kept = Counter()
    rows = []
    cache = {}
    sel = list(pos)
    for s in ("train", "val", "test"):
        ns = [n for n in neg if n["split"] == s]
        npos = sum(p["split"] == s for p in pos)
        sel += random.sample(ns, min(len(ns), int(npos * a.neg_ratio)))
    sel.sort(key=lambda it: it["af"])
    nvis = 0
    for it in sel:
        if it["af"] not in cache:
            cache.clear()
            cache[it["af"]] = imread(os.path.join(os.path.dirname(os.path.dirname(it["af"])), "img", it["img_name"]))
        img = cache[it["af"]]
        x1, y1, x2, y2 = it["box"]
        c = img[y1:y2, x1:x2].copy()
        if it["tmask"] is not None:
            c[~it["tmask"]] = 0
        h, w = c.shape[:2]
        fn = f"{it['stem']}_t{it['tooth']:02d}"
        cv2.imwrite(f"{a.out}/images/{it['split']}/{fn}.png", c)
        if it["polys"]:
            with open(f"{a.out}/labels/{it['split']}/{fn}.txt", "w") as f:
                for p in it["polys"]:
                    f.write("0 " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in p) + "\n")
        kept[(it["split"], "pos" if it["polys"] else "neg")] += 1
        rows.append([it["split"], fn, it["stem"], it["src"], it["fdi"], x1, y1, x2, y2, it["own"], len(it["polys"])])
        if nvis < a.vis and it["split"] == "train" and (it["polys"] or nvis % 4 == 0):
            v = c.copy()
            for p in it["polys"]:
                cv2.polylines(v, [p.astype(np.int32)], True, (0, 0, 255), 2)
            cv2.imwrite(f"{a.out}/vis/{fn}_{it['src']}.jpg", v); nvis += 1

    with open(f"{a.out}/meta.csv", "w", newline="", encoding="utf8") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "src_image", "tooth_src", "model_fdi", "x1", "y1", "x2", "y2", "own_lesions", "label_polys"])
        w.writerows(rows)
    with open(f"{a.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: caries\n")
    rep = {"stats": dict(st), "kept": {f"{s}_{t}": v for (s, t), v in sorted(kept.items())},
           "neg_pool": dict(Counter(n["split"] for n in neg))}
    json.dump(rep, open(f"{a.out}/report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
