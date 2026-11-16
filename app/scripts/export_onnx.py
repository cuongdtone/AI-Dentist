"""Chuyển YOLO26-seg (.pt) sang ONNX cho backend (onnxruntime CPU).
Dùng:
  python export_onnx.py --preset tooth  [--weights ...]   # -> ../backend/models/tooth_seg.onnx  (imgsz 1024)
  python export_onnx.py --preset caries [--weights ...]   # -> ../backend/models/caries_seg.onnx (imgsz 384)
Model được mount vào container -> export xong chỉ cần: docker compose restart backend
caries: chép thêm crop.json của dataset đã train (cách cắt răng) thành caries_seg.json cạnh file ONNX.
"""
import argparse, re, shutil
from pathlib import Path
from ultralytics import YOLO

RESEARCH = Path(__file__).resolve().parents[2] / "research"
PRESETS = {
    "tooth": dict(weights=RESEARCH / "tooth-detect/runs/segment/runs/ft_finaldata_30/weights/best.pt",
                  out="../backend/models/tooth_seg.onnx", imgsz=1024),
    # imgsz phải bằng imgsz lúc train (research/caries-seg/train.py), crop từng răng
    "caries": dict(weights=RESEARCH / "caries-seg/runs/segment/runs/caries_align_384/weights/best.pt",
                   out="../backend/models/caries_seg.onnx", imgsz=384),
}


def find_crop_json(weights):
    """weights .../runs/segment/runs/<name>/weights/best.pt -> args.yaml 'data:' -> <thư mục data>/crop.json."""
    run = Path(weights).resolve().parents[1]
    args = run / "args.yaml"
    if not args.exists():
        return None
    m = re.search(r"^data:\s*(.+)$", args.read_text(), re.M)
    if not m:
        return None
    data = Path(m.group(1).strip().strip("'\""))
    if not data.is_absolute():
        data = RESEARCH / "caries-seg" / data  # train.py chạy trong research/caries-seg
    c = data.parent / "crop.json"
    return c if c.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", choices=PRESETS, default="tooth")
    ap.add_argument("--weights", help="mặc định theo preset")
    ap.add_argument("--out", help="mặc định theo preset")
    ap.add_argument("--imgsz", type=int, help="mặc định theo preset")
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--crop-json", help="caries: crop.json của dataset (mặc định tự tìm theo args.yaml của run)")
    a = ap.parse_args()
    p = PRESETS[a.preset]
    weights, out, imgsz = a.weights or p["weights"], Path(a.out or p["out"]), a.imgsz or p["imgsz"]

    # Đầu ra (YOLO26 end-to-end, không cần NMS): output0 [1,300,6+32] = x1,y1,x2,y2,conf,cls,coef; output1 [1,32,H/4,W/4] = proto
    # Tên lớp được Ultralytics ghi vào metadata "names" của file ONNX.
    src = YOLO(str(weights)).export(format="onnx", imgsz=imgsz, opset=a.opset, simplify=True, dynamic=False, device="cpu")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        bak = out.with_suffix(".prev.onnx")
        shutil.copy2(out, bak)
        print(f"backup cũ -> {bak}")
    shutil.move(src, out)
    print(f"saved -> {out.resolve()}  (weights {weights}, imgsz {imgsz})")
    if a.preset == "caries":
        side = out.with_suffix(".json")
        crop = Path(a.crop_json) if a.crop_json else find_crop_json(weights)
        if crop:
            shutil.copy2(crop, side)
            print(f"cấu hình cắt răng {crop} -> {side}: {crop.read_text().strip()}")
        else:
            side.unlink(missing_ok=True)
            print("KHÔNG thấy crop.json (dataset cũ) -> app dùng cách cắt mặc định: bbox thẳng, lề 0.15")


if __name__ == "__main__":
    main()
