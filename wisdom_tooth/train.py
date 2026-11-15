"""Train YOLO26 phát hiện răng khôn hàm dưới lệch/ngầm trên phim X-quang toàn cảnh.
Dùng: python train.py --model yolo26s.pt --imgsz 1024 --epochs 150 --batch 8
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--model", default="yolo26s.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="yolo26s_640")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers,
        project="runs", name=a.name, exist_ok=True, cache="ram", patience=40, seed=0,
        cos_lr=True, close_mosaic=15,
        # Ảnh X-quang xám: tắt biến đổi màu, chỉ đổi độ sáng/tương phản nhẹ
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        # Răng khôn trái/phải cùng 1 lớp -> lật ngang hợp lệ; tránh lật dọc (đảo hàm trên/dưới)
        fliplr=0.5, flipud=0.0, degrees=5, translate=0.1, scale=0.3, mosaic=0.5, mixup=0.0,
    )
    best = YOLO(f"{model.trainer.save_dir}/weights/best.pt")
    for split in ("val", "test"):
        m = best.val(data=a.data, imgsz=a.imgsz, split=split, batch=a.batch, project="runs", name=f"{a.name}_{split}", exist_ok=True)
        print(f"[{split}] P={m.box.mp:.3f} R={m.box.mr:.3f} mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f}")


if __name__ == "__main__":
    main()
