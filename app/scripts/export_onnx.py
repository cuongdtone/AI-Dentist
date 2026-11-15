"""Chuyển YOLO26-seg (.pt) sang ONNX cho backend (onnxruntime CPU).
Dùng: python export_onnx.py \
        --weights ../../research/tooth-detect/runs/segment/runs/yolo26s-seg_1024/weights/best.pt \
        --out ../backend/models/tooth_seg.onnx
"""
import argparse, shutil
from pathlib import Path
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="../../research/tooth-detect/runs/segment/runs/yolo26s-seg_1024/weights/best.pt")
    ap.add_argument("--out", default="../backend/models/tooth_seg.onnx")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()

    # Đầu ra (YOLO26 end-to-end, không cần NMS): output0 [1,300,6+32] = x1,y1,x2,y2,conf,cls,coef; output1 [1,32,H/4,W/4] = proto
    # Tên lớp (FDI) được Ultralytics ghi vào metadata "names" của file ONNX.
    src = YOLO(a.weights).export(format="onnx", imgsz=a.imgsz, opset=a.opset, simplify=True, dynamic=False, device="cpu")
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(src, out)
    print(f"saved -> {out.resolve()}")


if __name__ == "__main__":
    main()
