"""Tạo dataset YOLO-seg vùng sâu răng (1 lớp 'caries') trên ảnh CẮT TỪNG RĂNG từ final_data.
Dùng: python prepare_data.py --out data_align            # mặc định: xoay thẳng răng, nhãn chỉ của răng giữa
      python prepare_data.py --out data --no-align --label all   # cách cũ (bbox thẳng, nhãn mọi vùng sâu trong khung)

Mỗi ảnh final_data:
  - Răng = viền 'Truc rang' của bác sĩ; răng bác sĩ không vẽ viền -> mask model FDI (conf >= --tooth-conf).
  - Vùng sâu = polygon 'Sau rang'; thuộc răng nào = răng chứa >= 50% diện tích của nó.
  - Cắt răng bằng crop_utils.crop_tooth (cùng hàm app dùng): --align xoay trục dài răng thẳng đứng rồi cắt sát.
  - --label own: chỉ vùng sâu CỦA RĂNG GIỮA KHUNG (răng kề có sâu vẫn là nền -> model học chỉ soi răng giữa).
    --label all: mọi vùng sâu lọt khung (>= --min-lesion-frac diện tích).
  - Răng có nhãn = mẫu dương; còn lại = mẫu âm, lấy tối đa --neg-ratio x số dương mỗi split.
  - Chỉ dùng ảnh bác sĩ CÓ vẽ 'Sau rang' (ảnh không vẽ có thể là chưa khám sâu răng).
Split theo wisdom_tooth/data. Cấu hình cắt ghi vào <out>/crop.json (export_onnx.py chép sang app).
Ra: <out>/{images,labels}/{train,val,test}, data.yaml, crop.json, meta.csv, report.json, vis/
"""
import argparse, csv, glob, json, os, random, shutil, sys
from collections import Counter
import cv2
import numpy as np
from ultralytics import YOLO
from crop_utils import crop_tooth, to_crop

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
    """Polygon răng của model FDI, bỏ phát hiện trùng (như app/backend model._dedupe)."""
    r = model.predict(img, imgsz=imgsz, conf=0.25, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return []
    M = r.masks.data.cpu().numpy() > 0.5
    S = r.boxes.conf.cpu().numpy()
    keep = []
    for i in np.argsort(-S):
        if S[i] < conf or M[i].sum() == 0:
            continue
        if all((M[i] & M[j]).sum() / min(M[i].sum(), M[j].sum()) < ios_thr for j in keep):
            keep.append(i)
    out = []
    for i in keep:
        cs, _ = cv2.findContours(M[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cs:
            c = max(cs, key=cv2.contourArea).reshape(-1, 2).astype(float)
            if len(c) >= 3:
                out.append((M[i], c))
    return out


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
    ap.add_argument("--out", default=os.path.join(ROOT, "data_align"))
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--tooth-conf", type=float, default=0.5)
    ap.add_argument("--align", action=argparse.BooleanOptionalAction, default=True, help="xoay trục răng thẳng đứng trước khi cắt")
    ap.add_argument("--margin", type=float, default=0.1, help="lề quanh răng (tỉ lệ kích thước răng sau khi xoay)")
    ap.add_argument("--mask-outside", action="store_true", help="tô đen ngoài mask răng (mặc định tắt)")
    ap.add_argument("--mask-dilate", type=int, default=6, help="nới mask răng (px) khi --mask-outside")
    ap.add_argument("--label", choices=("own", "all"), default="own")
    ap.add_argument("--neg-ratio", type=float, default=1.0, help="số mẫu âm / mẫu dương mỗi split")
    ap.add_argument("--min-lesion-frac", type=float, default=0.3, help="--label all: phần vùng sâu lọt khung tối thiểu")
    ap.add_argument("--vis", type=int, default=40)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    crop_cfg = {"align": a.align, "margin": a.margin, "mask_outside": a.mask_outside, "mask_dilate": a.mask_dilate}

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
        teeth = []
        for o in objs:
            if o["classTitle"] == "Truc rang" and o["geometryType"] == "line" and len(o["points"]["exterior"]) >= 3:
                p = np.array(o["points"]["exterior"], float)
                m = fill(p, H, W)
                if m.sum() > 0:
                    teeth.append((m, p, "doc"))
        for m, p in model_teeth(model, img, a.imgsz, a.tooth_conf):
            if all((m & d).sum() / m.sum() < 0.3 for d, _, _ in teeth):
                teeth.append((m, p, "model"))
        L = [fill(p, H, W) for p in les]
        larea = [max(int(l.sum()), 1) for l in L]
        owner = [-1] * len(L)
        for i, l in enumerate(L):
            ov = [(l & tm).sum() / larea[i] for tm, _, _ in teeth]
            if ov and max(ov) >= 0.5:
                owner[i] = int(np.argmax(ov))
        st["vùng sâu"] += len(L)
        st["vùng sâu không thuộc răng nào"] += owner.count(-1)
        for ti, (tm, tp, src) in enumerate(teeth):
            r = crop_tooth(img, tp, crop_cfg)
            if r is None:
                continue
            crop, M, ctm = r
            h, w = crop.shape[:2]
            own = [i for i in range(len(L)) if owner[i] == ti]
            cand = own if a.label == "own" else range(len(L))
            keep = np.zeros((h, w), np.uint8)
            for i in cand:
                lm = np.zeros((h, w), np.uint8)
                cv2.fillPoly(lm, [np.round(to_crop(les[i], M)).astype(np.int32)], 1)
                if a.mask_outside:
                    d = a.mask_dilate
                    lm &= cv2.dilate(ctm.astype(np.uint8), np.ones((2 * d + 1, 2 * d + 1), np.uint8))
                if i in own or lm.sum() >= a.min_lesion_frac * larea[i]:
                    keep |= lm
            item = dict(split=s, stem=stem, tooth=ti, src=src, own=len(own), polys=mask_polys(keep), crop=crop, tpoly=to_crop(tp, M))
            if item["polys"]:
                pos.append(item)
                if not own:
                    st["răng lành nhưng khung có sâu răng kề (label all)"] += 1
            else:
                item["crop"] = None  # mẫu âm: chỉ giữ ảnh nếu được chọn (tiết kiệm RAM)
                item["af"], item["name"], item["poly"] = af, name, tp
                neg.append(item)
        st["răng nguồn doc"] += sum(t[2] == "doc" for t in teeth)
        st["răng nguồn model"] += sum(t[2] == "model" for t in teeth)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(anns)}...", flush=True)

    random.seed(0)
    sel = list(pos)
    for s in ("train", "val", "test"):
        ns = [n for n in neg if n["split"] == s]
        sel += random.sample(ns, min(len(ns), int(sum(p["split"] == s for p in pos) * a.neg_ratio)))
    kept, rows, nvis, cache = Counter(), [], 0, {}
    for it in sel:
        if it["crop"] is None:
            if it["af"] not in cache:
                cache.clear()
                cache[it["af"]] = imread(os.path.join(os.path.dirname(os.path.dirname(it["af"])), "img", it["name"]))
            it["crop"], M, _ = crop_tooth(cache[it["af"]], it["poly"], crop_cfg)
            it["tpoly"] = to_crop(it["poly"], M)
        c = it["crop"]
        h, w = c.shape[:2]
        fn = f"{it['stem']}_t{it['tooth']:02d}"
        cv2.imwrite(f"{a.out}/images/{it['split']}/{fn}.png", c)
        if it["polys"]:
            with open(f"{a.out}/labels/{it['split']}/{fn}.txt", "w") as f:
                for p in it["polys"]:
                    f.write("0 " + " ".join(f"{x / w:.6f} {y / h:.6f}" for x, y in p) + "\n")
        kept[(it["split"], "pos" if it["polys"] else "neg")] += 1
        rows.append([it["split"], fn, it["stem"], it["src"], w, h, it["own"], len(it["polys"])])
        if nvis < a.vis and it["split"] == "train" and (it["polys"] or nvis % 4 == 0):
            v = c.copy()
            cv2.polylines(v, [np.round(it["tpoly"]).astype(np.int32)], True, (0, 255, 0), 1)
            for p in it["polys"]:
                cv2.polylines(v, [p.astype(np.int32)], True, (0, 0, 255), 2)
            cv2.imwrite(f"{a.out}/vis/{fn}_{it['src']}.jpg", v); nvis += 1
        it["crop"] = None

    with open(f"{a.out}/meta.csv", "w", newline="", encoding="utf8") as f:
        wr = csv.writer(f)
        wr.writerow(["split", "file", "src_image", "tooth_src", "w", "h", "own_lesions", "label_polys"])
        wr.writerows(rows)
    with open(f"{a.out}/data.yaml", "w") as f:
        f.write(f"path: {os.path.abspath(a.out)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: caries\n")
    json.dump({**crop_cfg, "label": a.label}, open(f"{a.out}/crop.json", "w"), indent=1)
    rep = {"crop": crop_cfg, "label": a.label, "stats": dict(st),
           "kept": {f"{s}_{t}": v for (s, t), v in sorted(kept.items())},
           "neg_pool": dict(Counter(n["split"] for n in neg))}
    json.dump(rep, open(f"{a.out}/report.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
