"""Phân tích răng khôn trên ảnh panoramic bất kỳ: FDI-seg -> Winter, độ sâu, ngầm/lệch.
Dùng: python analyze.py --source anh.png|thu_muc --out out_analyze
Ghi: <out>/<tên>.jpg (ảnh vẽ trục, mức mặt nhai/CEJ) và <out>/<tên>.json.
"""
import argparse, glob, json, os, sys
import cv2
from ultralytics import YOLO
from wisdom_geom import analyze_teeth, describe, draw, load_params, predict_masks

sys.stdout.reconfigure(encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--weights", default="../tooth-detect/runs/segment/runs/yolo26s-seg_1024/weights/best.pt")
    ap.add_argument("--params", default="params.json")
    ap.add_argument("--out", default="out_analyze")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()

    p = load_params(a.params)
    model = YOLO(a.weights)
    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.source, "*"))) if os.path.isdir(a.source) else [a.source]
    for f in files:
        img = cv2.imread(f)
        if img is None:
            continue
        masks = predict_masks(model, img, a.imgsz, a.conf, a.device)
        res = analyze_teeth(masks, p)
        stem = os.path.splitext(os.path.basename(f))[0]
        cv2.imwrite(os.path.join(a.out, stem + ".jpg"), draw(img, res, masks))
        json.dump(res, open(os.path.join(a.out, stem + ".json"), "w"), ensure_ascii=False, indent=1)
        print(f"{os.path.basename(f)}:")
        for r in res:
            print("   " + describe(r))
        if not res:
            print("   không thấy răng khôn")


if __name__ == "__main__":
    main()
