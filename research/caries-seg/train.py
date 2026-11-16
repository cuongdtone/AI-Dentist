"""Train YOLO26-seg bắt vùng sâu răng (1 lớp) trên ảnh cắt từng răng (từ prepare_data.py).
Dùng: python train.py --data data_align/data.yaml --name caries_align_384 --epochs 100 --batch 16 --workers 0
Crop xoay thẳng + nhãn chỉ của răng giữa -> để --mosaic 0 (mosaic ghép 4 răng, phá quy tắc "răng giữa khung").
"""
import argparse
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_align/data.yaml")
    ap.add_argument("--model", default="../tooth-detect/yolo26s-seg.pt")
    ap.add_argument("--imgsz", type=int, default=384, help="crop răng ~150-400px; vùng sâu nhỏ nên đừng hạ thấp")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--mosaic", type=float, default=0.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--fraction", type=float, default=1.0)
    a = ap.parse_args()
    name = a.name or f"caries_align_{a.imgsz}"

    model = YOLO(a.model)
    model.train(
        data=a.data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, workers=a.workers, fraction=a.fraction,
        project="runs", name=name, exist_ok=True, cache="disk", patience=30, seed=0, cos_lr=True,
        close_mosaic=15, overlap_mask=False,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,           # X-quang xám
        fliplr=0.5, flipud=0.0,                    # 1 lớp -> lật ngang an toàn; lật dọc đảo hàm trên/dưới
        degrees=5, translate=0.05, scale=0.2,  # nhẹ: răng đã xoay thẳng, giữ ở giữa khung
        mosaic=a.mosaic, mixup=0.0,
    )
    best = YOLO(f"{model.trainer.save_dir}/weights/best.pt")
    for split in ("val", "test"):
        m = best.val(data=a.data, imgsz=a.imgsz, batch=a.batch, split=split, project="runs", name=f"{name}_{split}", exist_ok=True)
        print(f"{split}  box mAP50={m.box.map50:.3f} | mask mAP50={m.seg.map50:.3f} mAP50-95={m.seg.map:.3f} "
              f"| P={m.seg.mp:.3f} R={m.seg.mr:.3f}")


if __name__ == "__main__":
    main()
