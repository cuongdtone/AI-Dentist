"""So sánh các model FDI-seg trên nhiều tập.
Dùng: python compare_models.py --weights fdi_v1_dentex.pt runs/segment/runs/ft_finaldata_30/weights/best.pt

1. mAP (box/mask) trên val (DENTEX + final_data) và test (fd_, 50 ảnh final_data).
   Lưu ý: nhãn fd_ có số FDI do model v1 gán -> hơi thiên vị v1; viền răng là của bác sĩ.
2. Tỉ lệ ảnh có số FDI trùng trên 134 ảnh test của wisdom_tooth (final_data, KHÔNG cần nhãn,
   không model nào được train trên các ảnh này).
"""
import argparse, glob, os, sys
from collections import Counter
from ultralytics import YOLO

sys.stdout.reconfigure(encoding="utf-8")


def dup_rate(model, files, imgsz):
    n_dup = n_teeth = 0
    for f in files:
        r = model.predict(f, imgsz=imgsz, conf=0.25, verbose=False)[0]
        c = Counter(model.names[int(k)] for k in r.boxes.cls)
        n_teeth += sum(c.values())
        n_dup += any(v > 1 for v in c.values())
    return n_dup / max(len(files), 1), n_teeth / max(len(files), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--holdout", default="../../wisdom_tooth/data/images/test")
    ap.add_argument("--imgsz", type=int, default=1024)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.holdout, "*")))
    rows = []
    for w in a.weights:
        m = YOLO(w)
        res = {"model": w}
        for split in ("val", "test"):
            v = m.val(data=a.data, split=split, imgsz=a.imgsz, batch=4, plots=False, verbose=False,
                      project="runs/compare", name="tmp", exist_ok=True)
            res[f"{split}_box50"], res[f"{split}_mask50"], res[f"{split}_mask5095"] = v.box.map50, v.seg.map50, v.seg.map
        res["dup_rate"], res["teeth_per_img"] = dup_rate(m, files, a.imgsz)
        rows.append(res)
    print(f"\n{'model':60s} {'val box50':>9} {'val mask50':>10} {'val m50-95':>10} {'test(fd) mask50':>15} {'ảnh trùng FDI':>13} {'răng/ảnh':>8}")
    for r in rows:
        print(f"{r['model'][-60:]:60s} {r['val_box50']:9.3f} {r['val_mask50']:10.3f} {r['val_mask5095']:10.3f} "
              f"{r['test_mask50']:15.3f} {r['dup_rate']:13.1%} {r['teeth_per_img']:8.1f}")
    print(f"(ảnh trùng FDI đo trên {len(files)} ảnh holdout final_data)")


if __name__ == "__main__":
    main()
