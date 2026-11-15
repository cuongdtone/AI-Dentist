"""Train YOLO26-seg nhận diện + đánh số răng FDI (32 lớp) trên phim panoramic DENTEX.
Dùng: python train.py --model yolo26s-seg.pt --imgsz 1024 --epochs 150 --batch 4
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--model", default="yolo26s-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fraction", type=float, default=1.0, help="<1 để chạy thử nhanh")
    a = ap.parse_args()
    name = a.name or f"{a.model.split('.')[0]}_{a.imgsz}"

    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers, fraction=a.fraction,
        project="runs", name=name, exist_ok=True, cache="disk", patience=40, seed=0,
        cos_lr=True, close_mosaic=20, overlap_mask=False,  # răng chạm nhau -> mask riêng từng răng
        # Ảnh X-quang xám: tắt biến đổi màu, chỉ đổi độ sáng/tương phản nhẹ
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
        # BẮT BUỘC fliplr=0: lật ngang đổi trái/phải nhưng Ultralytics không đổi lớp (11<->21...).
        # Bản lật đúng nhãn đã được tạo offline trong prepare_data.py. flipud cũng sai (đảo hàm trên/dưới).
        fliplr=0.0, flipud=0.0,
        # Đánh số phụ thuộc ngữ cảnh toàn hàm -> mosaic/scale vừa phải, xoay nhẹ
        degrees=5, translate=0.1, scale=0.25, mosaic=0.5, mixup=0.0,
    )
    best = YOLO(f"{model.trainer.save_dir}/weights/best.pt")
    m = best.val(data=a.data, imgsz=a.imgsz, batch=a.batch, split="val", project="runs", name=f"{name}_val", exist_ok=True)
    print(f"val  box mAP50={m.box.map50:.3f} mAP50-95={m.box.map:.3f} | mask mAP50={m.seg.map50:.3f} mAP50-95={m.seg.map:.3f}")


if __name__ == "__main__":
    main()
