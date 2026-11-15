"""Dự đoán răng khôn trên ảnh/thư mục ảnh.
Dùng: python predict.py --weights runs/yolo26s_1024/weights/best.pt --source anh.png --conf 0.25
Kết quả ảnh vẽ box lưu tại runs/predict/.
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/detect/runs/yolo26s_1024/weights/best.pt")
    ap.add_argument("--source", required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    a = ap.parse_args()

    model = YOLO(a.weights)
    for r in model.predict(a.source, imgsz=a.imgsz, conf=a.conf, save=True, project="runs", name="predict", exist_ok=True, stream=True):
        print(f"{r.path}: {len(r.boxes)} răng khôn")
        for b in r.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            print(f"   conf={b.conf.item():.2f}  box=({x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f})")


if __name__ == "__main__":
    main()
