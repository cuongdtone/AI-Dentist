# e2e-evaluation — đánh giá end-to-end pipeline webapp trên final_data

`evaluate.py` chạy **đúng code backend của app** (`app/backend/app`: `ToothSegmenter` ONNX → `analyze_wisdom` → `analyze_caries`, config mặc định) trên ảnh `final_data`, so với nhãn bác sĩ.

- **Model**: luôn lấy `best.pt` mới nhất (mtime) trong `research/tooth-detect/runs` và `research/caries-seg/runs`, export ONNX vào `cache/onnx/` (export lại khi `.pt` đổi). Caries: chép `crop.json` của dataset đã train → cách cắt răng giống app. Chỉ định tay: `--tooth-weights`, `--caries-weights`.
- **Split** theo `wisdom_tooth/data`: `test` không dùng để train model răng lẫn sâu răng → **số liệu chính**.
- Dự đoán cache trong `cache/pred/<model>/` (đổi model → cache mới; `--no-cache` để chạy lại).

```bash
P=../../wisdom_tooth/env/bin/python
$P evaluate.py                       # toàn bộ final_data
$P evaluate.py --splits test         # chỉ test
$P evaluate.py --limit 10 --no-vis   # thử nhanh
```

Ra `out/<tooth_run>__<caries_run>/`:

| File | Nội dung |
|---|---|
| `vis/<split>/<doc>_<ảnh>.jpg` | trái: nhãn bác sĩ (viền răng xám, vùng sâu đỏ, box răng khôn lệch ngầm) · phải: dự đoán, đồ hoạ như webapp (✗ = vùng sâu không khớp nhãn) |
| `report.md` / `report.json` | precision / recall / F1 / specificity / accuracy theo split |
| `wisdom_teeth.csv` | từng răng 38/48: nhãn, dự đoán, góc Winter, độ sâu, cờ |
| `caries_teeth.csv`, `caries_lesions.csv` | sâu răng theo răng / theo vùng |

Cách tính:
- **Răng khôn**: răng 38/48 app phân tích được; nhãn = 1 nếu mask khớp box *Rang khon ham duoi lech ngam* (IoU ≥ `--wis-iou` hoặc tâm mask trong box). Nhãn chỉ có hàm dưới.
- **Sâu răng mức răng**: vùng sâu bác sĩ gán cho răng app tìm thấy giao nhiều nhất; dự đoán = răng có ≥ 1 vùng sâu.
- **Sâu răng mức vùng**: ghép 1-1 theo IoU polygon ≥ `--car-iou`.
- Báo cáo riêng *ảnh bác sĩ có vẽ sâu răng* (như tập train) và *toàn bộ ảnh* (coi ảnh không vẽ là không sâu).
