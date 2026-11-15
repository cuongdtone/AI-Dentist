# wisdom-analysis — phân tích răng khôn (giai đoạn 1, không cần bác sĩ gán nhãn)

Model FDI-seg (`../tooth-detect`) → mask răng 7 & 8 → hình học:
- **Winter**: góc trục răng khôn so với trục răng 7 (dương = nghiêng gần).
- **Độ sâu A/B/C** (Pell & Gregory / Archer): đỉnh răng khôn so với mặt nhai và CEJ (ước lượng) của răng 7.
- **Ngầm/lệch**: `depth_ratio > impact_depth` HOẶC `|góc| > angle_thr`, ngưỡng chỉnh theo nhãn *Impacted* của DENTEX.

| File | Việc |
|---|---|
| `wisdom_geom.py` | lõi hình học + vẽ |
| `extract_features.py` | chạy model lên DENTEX train (705 ảnh bệnh) / test (250) → `features_*.csv` |
| `calibrate.py` | grid search ngưỡng trên train, báo cáo test → `params.json` |
| `analyze.py` | phân tích ảnh bất kỳ → ảnh vẽ + JSON |
| `make_review.py` | trang HTML duyệt Winter thủ công + chấm điểm |

```bash
P=../../wisdom_tooth/env/bin/python
$P extract_features.py --split train && $P extract_features.py --split test
$P calibrate.py
$P analyze.py --source anh.png --out out_analyze
$P make_review.py --n 120          # mở review/index.html, duyệt, Tải kết quả
$P make_review.py --score ~/Downloads/review_result.json
```

Lưu ý DENTEX: các thư mục con là **bộ ảnh khác nhau dù trùng tên file** → không ghép nhãn theo tên.
