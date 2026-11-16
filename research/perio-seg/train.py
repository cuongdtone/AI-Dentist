"""Train YOLO26-seg vùng thân răng (CEJ) + vùng giữa đỉnh xương (mức xương nha chu) trên phim panoramic.
Dùng: python train.py --epochs 100 --batch 4 --workers 0
3 lớp không phụ thuộc trái/phải -> lật ngang an toàn; KHÔNG lật dọc (đảo hàm trên/dưới).
Mỗi ảnh chỉ 3 vùng lớn, chồng lên nhau (thân răng nằm trong vùng xương) -> overlap_mask=False.
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data.yaml")
    ap.add_argument("--model", default="../tooth-detect/yolo26s-seg.pt")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--mosaic", type=float, default=0.0, help="vùng theo cả cung hàm -> mosaic cắt mất ngữ cảnh")
    ap.add_argument("--name", default=None)
    ap.add_argument("--fraction", type=float, default=1.0, help="<1 để chạy thử nhanh")
    ap.add_argument("--project", default="runs")
    a = ap.parse_args()
    name = a.name or f"perio_{a.imgsz}"

    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers, fraction=a.fraction,
        project=a.project, name=name, exist_ok=True, cache="disk", patience=30, seed=0, cos_lr=True,
        overlap_mask=False, mask_ratio=2,          # mask độ phân giải cao hơn mặc định (4): mép CEJ / xương cần sát
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,           # X-quang xám
        fliplr=0.5, flipud=0.0,
        degrees=3, translate=0.05, scale=0.15, mosaic=a.mosaic, mixup=0.0,
    )
    best = YOLO(f"{model.trainer.save_dir}/weights/best.pt")
    for split in ("val", "test"):
        m = best.val(data=a.data, imgsz=a.imgsz, batch=a.batch, split=split, project=a.project, name=f"{name}_{split}", exist_ok=True)
        print(f"{split}  box mAP50={m.box.map50:.3f} | mask mAP50={m.seg.map50:.3f} mAP50-95={m.seg.map:.3f}")


if __name__ == "__main__":
    main()
