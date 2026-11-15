"""Gộp final_data (ảnh + viền răng bác sĩ, KHÔNG có số FDI) vào dataset YOLO-seg FDI (data/).
Dùng: python add_final_data.py --weights runs/segment/runs/yolo26s-seg_1024/weights/best.pt

Nhãn cho mỗi ảnh (pseudo-label có kiểm soát):
  1. Viền 'Truc rang' của bác sĩ (polyline khép kín) khớp mask model (IoU >= --match) -> polygon BÁC SĨ + số FDI của MODEL.
  2. Răng model thấy rõ (conf >= --pseudo-conf) mà bác sĩ không vẽ viền (hay gặp: răng khôn) -> polygon MODEL.
  3. Viền bác sĩ không khớp model -> suy số FDI từ khe trống giữa 2 răng kề cùng phần tư; không suy được -> bỏ ảnh.
  4. Kiểm tra: trong mỗi phần tư, số răng phải tăng dần theo khoảng cách tới đường giữa, không trùng -> sai thì bỏ ảnh.
'Binh thuong' trùng 'Truc rang' nên bỏ qua. Chia train/val/test theo wisdom_tooth/data (test KHÔNG đưa vào train/val).
Ảnh được ghi với tiền tố fd_ (chạy lại sẽ xoá fd_* cũ); train có thêm bản lật ngang đổi lớp như prepare_data.py.
"""
import argparse, glob, json, os, sys
from collections import Counter
import cv2
import numpy as np
from ultralytics import YOLO
from prepare_data import cls_idx, flip_cls, write_sample

sys.stdout.reconfigure(encoding="utf-8")


def fdi_to_cls(f):
    return cls_idx(f // 10, f % 10)


def split_names(wdir):
    out = {}
    for s in ("train", "val", "test"):
        for f in os.listdir(os.path.join(wdir, "images", s)):
            out[os.path.splitext(f)[0]] = s
    return out


def poly_mask(p, H, W):
    m = np.zeros((H, W), np.uint8)
    cv2.fillPoly(m, [np.round(p).astype(np.int32)], 1)
    return m.astype(bool)


def mask_poly(m):
    cs, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 1.5, True).reshape(-1, 2).astype(float)
    return c if len(c) >= 3 else None


def label_image(img, doc_polys, model, a):
    """-> (list[(fdi, polygon)], lý do bỏ hoặc None, thống kê)"""
    H, W = img.shape[:2]
    r = model.predict(img, imgsz=a.imgsz, conf=0.25, retina_masks=True, verbose=False)[0]
    if r.masks is None:
        return None, "model không thấy răng", {}
    P = r.masks.data.cpu().numpy() > 0.5
    C = [int(model.names[int(c)]) for c in r.boxes.cls.cpu().numpy()]
    S = r.boxes.conf.cpu().numpy()
    D = [poly_mask(p, H, W) for p in doc_polys]
    teeth, used, unmatched, st = [], set(), [], Counter()
    for p, d in zip(doc_polys, D):
        ious = [(d & q).sum() / max((d | q).sum(), 1) for q in P]
        j = int(np.argmax(ious))
        if ious[j] >= a.match and j not in used:
            used.add(j); teeth.append([C[j], p, "doc"]); st["doc"] += 1
        else:
            unmatched.append((p, d))
    for j in range(len(P)):
        if j in used or S[j] < a.pseudo_conf:
            continue
        # bỏ nếu chồng lên viền bác sĩ đã dùng (trùng phát hiện)
        if any((P[j] & d).sum() / max(P[j].sum(), 1) > 0.3 for d in D):
            continue
        p = mask_poly(P[j])
        if p is not None:
            teeth.append([C[j], p, "model"]); st["model"] += 1

    def cen(p):
        return np.asarray(p, float).mean(0)
    mids = [cen(p)[0] for f, p, _ in teeth if f % 10 in (1, 2)]
    mid = float(np.median(mids)) if mids else W / 2

    # 3. suy số cho viền bác sĩ không khớp: chèn vào khe trống của phần tư gần nhất
    for p, d in unmatched:
        c = cen(p)
        near = min(teeth, key=lambda t: np.linalg.norm(cen(t[1]) - c), default=None)
        if near is None:
            return None, "viền không suy được số", st
        upper = near[0] // 10 in (1, 2)
        left = c[0] < mid
        q = (1 if left else 2) if upper else (4 if left else 3)
        same = sorted([(abs(cen(tp)[0] - mid), f % 10) for f, tp, _ in teeth if f // 10 == q])
        dist = abs(c[0] - mid)
        before = [n for x, n in same if x < dist]; after = [n for x, n in same if x > dist]
        lo = before[-1] if before else 0; hi = after[0] if after else 9
        if hi - lo == 2:
            n = lo + 1
        elif not after and lo < 8 and lo >= 1:
            n = lo + 1
        elif not before and hi > 1 and hi <= 8 and hi - 1 >= 1:
            n = hi - 1
        else:
            return None, "viền không suy được số", st
        teeth.append([q * 10 + n, p, "doc_infer"]); st["doc_infer"] += 1

    # 4. kiểm tra thứ tự & trùng
    cnt = Counter(f for f, _, _ in teeth)
    if any(v > 1 for v in cnt.values()):
        return None, "trùng số FDI", st
    for q in (1, 2, 3, 4):
        seq = [f % 10 for _, f in sorted((abs(cen(p)[0] - mid), f) for f, p, _ in teeth if f // 10 == q)]
        if seq != sorted(seq):
            return None, "sai thứ tự trong phần tư", st
    return [(f, np.asarray(p, float)) for f, p, _ in teeth], None, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/segment/runs/yolo26s-seg_1024/weights/best.pt")
    ap.add_argument("--src", default="../../final_data")
    ap.add_argument("--split-from", default="../../wisdom_tooth/data")
    ap.add_argument("--out", default="data")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--match", type=float, default=0.4)
    ap.add_argument("--pseudo-conf", type=float, default=0.5)
    ap.add_argument("--max-side", type=int, default=1600)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--vis", type=int, default=0, help="lưu N ảnh kiểm tra nhãn vào <out>/fd_vis")
    a = ap.parse_args()

    split = split_names(a.split_from)
    for s in ("train", "val", "test"):
        for d in ("images", "labels"):
            os.makedirs(f"{a.out}/{d}/{s}", exist_ok=True)
            for f in glob.glob(f"{a.out}/{d}/{s}/fd_*"):
                os.remove(f)
    if a.vis:
        os.makedirs(f"{a.out}/fd_vis", exist_ok=True)
    model = YOLO(a.weights)
    reasons, stats, kept, seen = Counter(), Counter(), Counter(), set()
    anns = sorted(glob.glob(f"{a.src}/*/ann/*.json"))[: a.limit or None]
    for k, af in enumerate(anns):
        doc = af.split(os.sep)[-3]; name = os.path.basename(af)[:-5]
        stem = f"{doc}_{os.path.splitext(name)[0]}"
        s = split.get(stem)
        if s is None:
            reasons["không có trong split wisdom_tooth"] += 1; continue
        ann = json.load(open(af, encoding="utf8"))
        polys = [np.array(o["points"]["exterior"], float) for o in ann["objects"]
                 if o["classTitle"] == "Truc rang" and o["geometryType"] == "line" and len(o["points"]["exterior"]) >= 3]
        if len(polys) < 10:
            reasons["không có viền Truc rang"] += 1; continue
        img = cv2.imdecode(np.fromfile(os.path.join(os.path.dirname(os.path.dirname(af)), "img", name), np.uint8), cv2.IMREAD_COLOR)
        teeth, why, st = label_image(img, polys, model, a)
        if why:
            reasons[why] += 1; continue
        stats.update(st)
        H, W = img.shape[:2]
        sc = min(1.0, a.max_side / max(H, W))
        objs = [(fdi_to_cls(f), p) for f, p in teeth]
        write_sample(img, objs, W, H, f"{a.out}/images/{s}/fd_{stem}.png", f"{a.out}/labels/{s}/fd_{stem}.txt", sc)
        kept[s] += 1
        if s == "train":
            fobjs = [(flip_cls(c), np.stack([W - p[:, 0], p[:, 1]], 1)) for c, p in objs]
            write_sample(cv2.flip(img, 1), fobjs, W, H, f"{a.out}/images/{s}/fd_{stem}_flip.png", f"{a.out}/labels/{s}/fd_{stem}_flip.txt", sc)
        if a.vis and sum(kept.values()) <= a.vis:
            v = img.copy()
            for f, p in teeth:
                cv2.polylines(v, [p.astype(np.int32)], True, (0, 255, 0), 2)
                c = p.mean(0).astype(int); cv2.putText(v, str(f), tuple(c), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.imwrite(f"{a.out}/fd_vis/{stem}.jpg", v)
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(anns)}...", flush=True)
    print("Giữ:", dict(kept), "| Bỏ:", dict(reasons))
    print("Nguồn nhãn răng:", dict(stats))
    json.dump({"kept": kept, "dropped": reasons, "sources": stats}, open(f"{a.out}/fd_report.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
