"""Đo độ chính xác ĐÁNH SỐ FDI trên tập test DENTEX (250 ảnh, nhãn LabelMe).
Dùng: python eval_test.py --weights runs/segment/runs/yolo26s-seg_1024/weights/best.pt

Tập test chỉ gán nhãn các răng có bệnh/điều trị (label dạng "classIdx-tên-FDI", vd "1-çürük-26"),
không phải toàn bộ răng -> không tính mAP được. Thay vào đó, với mỗi răng có nhãn:
  - ghép với mask dự đoán có IoU lớn nhất (>= --iou)  -> "phát hiện được"
  - so số FDI dự đoán với FDI nhãn                   -> "đánh số đúng"
"""
import argparse, glob, json, os, sys
from collections import Counter
import cv2
import numpy as np
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--src", default="../dentex/extracted/disease")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.3, help="IoU tối thiểu giữa mask nhãn và mask dự đoán")
    ap.add_argument("--device", default=None, help="vd 0 hoặc cpu")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu (kiểm thử)")
    ap.add_argument("--save-errors", default=None, help="thư mục lưu ảnh vẽ các răng đánh số sai")
    a = ap.parse_args()

    model = YOLO(a.weights)
    names = model.names
    n = det = correct = 0
    off = Counter()  # sai lệch: (FDI nhãn, FDI dự đoán)
    per_q = Counter(); per_q_ok = Counter()
    if a.save_errors:
        os.makedirs(a.save_errors, exist_ok=True)

    files = sorted(glob.glob(os.path.join(a.src, "label/*.json")))
    for jf in files[:a.limit] if a.limit else files:
        lab = json.load(open(jf, encoding="utf8"))
        img_p = os.path.join(a.src, "input", os.path.basename(jf)[:-5] + ".png")
        img = cv2.imread(img_p); H, W = img.shape[:2]
        r = model.predict(img, imgsz=a.imgsz, conf=a.conf, retina_masks=True, device=a.device, verbose=False)[0]
        pm = r.masks.data.cpu().numpy().astype(bool) if r.masks is not None else np.zeros((0, H, W), bool)
        pc = [names[int(c)] for c in r.boxes.cls.cpu().numpy()] if len(r.boxes) else []
        errs = []
        for s in lab["shapes"]:
            fdi = s["label"].split("-")[-1]
            gm = np.zeros((H, W), np.uint8)
            cv2.fillPoly(gm, [np.array(s["points"], np.int32)], 1); gm = gm.astype(bool)
            n += 1; per_q[fdi[0]] += 1
            if len(pm) == 0:
                continue
            inter = (pm & gm).sum((1, 2)); union = (pm | gm).sum((1, 2))
            ious = inter / np.maximum(union, 1); j = int(ious.argmax())
            if ious[j] < a.iou:
                errs.append((s["points"], fdi, "miss")); continue
            det += 1
            if pc[j] == fdi:
                correct += 1; per_q_ok[fdi[0]] += 1
            else:
                off[(fdi, pc[j])] += 1; errs.append((s["points"], fdi, pc[j]))
        if a.save_errors and errs:
            vis = r.plot(boxes=False, labels=True, conf=False)
            for pts, g, p in errs:
                pts = np.array(pts, np.int32); cv2.polylines(vis, [pts], True, (0, 0, 255), 3)
                cv2.putText(vis, f"GT{g}/P{p}", tuple(pts.min(0)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            cv2.imwrite(os.path.join(a.save_errors, os.path.basename(img_p)[:-4] + ".jpg"), vis)

    print(f"Răng có nhãn: {n}")
    print(f"Phát hiện được (IoU>={a.iou}): {det} ({det / n:.1%})")
    print(f"Đánh số đúng / tổng: {correct} ({correct / n:.1%}) | đúng / phát hiện được: {correct / max(det, 1):.1%}")
    print("Theo quadrant (đúng/tổng):", {q: f"{per_q_ok[q]}/{per_q[q]}" for q in sorted(per_q)})
    print("Nhầm lẫn hay gặp (nhãn -> dự đoán):", off.most_common(15))


if __name__ == "__main__":
    main()
