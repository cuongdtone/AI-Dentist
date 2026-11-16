# Đánh giá end-to-end — ft_finaldata_30__caries_align_384

- Ngày chạy: 2026-11-16 10:53 · ảnh (bỏ trùng hash): 891 · thời gian: 1051s
- Model răng: `research/tooth-detect/runs/segment/runs/ft_finaldata_30/weights/best.pt`
- Model sâu răng: `research/caries-seg/runs/segment/runs/caries_align_384/weights/best.pt` (crop: {'align': True, 'margin': 0.1, 'mask_outside': False, 'mask_dilate': 6, 'label': 'own'})
- Pipeline: code backend của app, config mặc định (seg conf 0.25, caries conf 0.25, wisdom params mặc định).
- **Số liệu chính: split `test`** (không dùng để train model răng lẫn model sâu răng). `val`/`train` đã thấy lúc train → lạc quan.

## Răng khôn ngầm/lệch (hàm dưới 38/48)

Đơn vị: răng 38/48 app tìm thấy và phân tích được. Nhãn = 1 nếu mask khớp box *Rang khon ham duoi lech ngam* (IoU ≥ 0.3 hoặc tâm mask nằm trong box). Giả định bác sĩ khoanh đủ mọi răng khôn hàm dưới lệch ngầm.

| Split | Precision | Recall | F1 | Specificity | Accuracy | TP / FP / FN / TN |
|---|---|---|---|---|---|---|
| test | 0.591 | 0.793 | 0.677 | 0.579 | 0.672 | 65 / 45 / 17 / 62 |
| val | 0.583 | 0.778 | 0.667 | 0.545 | 0.650 | 63 / 45 / 18 / 54 |
| train | 0.559 | 0.781 | 0.651 | 0.532 | 0.640 | 286 / 226 / 80 / 257 |
| all | 0.567 | 0.783 | 0.658 | 0.541 | 0.646 | 414 / 316 / 115 / 373 |

| Split | Box bác sĩ | Có răng app khớp | App kết luận ngầm/lệch (recall theo box) |
|---|---|---|---|
| test | 84 | 82 | 65 (77.4%) |
| val | 83 | 81 | 63 (75.9%) |
| train | 383 | 366 | 286 (74.7%) |
| all | 550 | 529 | 414 (75.3%) |

Theo răng (test):

| Split | Precision | Recall | F1 | Specificity | Accuracy | TP / FP / FN / TN |
|---|---|---|---|---|---|---|
| 38 | 0.617 | 0.787 | 0.692 | 0.540 | 0.660 | 37 / 23 / 10 / 27 |
| 48 | 0.560 | 0.800 | 0.659 | 0.614 | 0.685 | 28 / 22 / 7 / 35 |

## Sâu răng — ảnh bác sĩ có vẽ sâu răng

Chỉ ảnh có ≥ 1 polygon *Sau rang (polygon)* (giống tập train: ảnh không vẽ sâu răng có thể là chưa gán nhãn).

**Mức răng** — đơn vị: răng app tìm thấy; nhãn = 1 nếu có vùng sâu bác sĩ giao nhiều nhất với răng đó; dự đoán = 1 nếu app báo ≥ 1 vùng sâu.

| Split | Precision | Recall | F1 | Specificity | Accuracy | TP / FP / FN / TN |
|---|---|---|---|---|---|---|
| test | 0.468 | 0.807 | 0.593 | 0.896 | 0.887 | 213 / 242 / 51 / 2085 |
| val | 0.538 | 0.870 | 0.665 | 0.904 | 0.901 | 220 / 189 / 33 / 1791 |
| train | 0.552 | 0.902 | 0.685 | 0.914 | 0.913 | 1077 / 874 / 117 / 9333 |
| all | 0.536 | 0.882 | 0.667 | 0.910 | 0.907 | 1510 / 1305 / 201 / 13209 |

**Mức vùng** — ghép 1-1 tham lam theo IoU polygon ≥ 0.2.

| Split | Ảnh | Vùng GT | Vùng dự đoán | Precision | Recall | F1 | GT không thuộc răng nào |
|---|---|---|---|---|---|---|---|
| test | 93 | 295 | 486 | 0.432 | 0.712 | 0.538 | 4 |
| val | 80 | 282 | 429 | 0.513 | 0.780 | 0.619 | 13 |
| train | 410 | 1327 | 2083 | 0.529 | 0.830 | 0.646 | 47 |
| all | 583 | 1904 | 2998 | 0.511 | 0.805 | 0.625 | 64 |

**Mức ảnh** — có / không có sâu răng.

| Split | Precision | Recall | F1 | Specificity | Accuracy | TP / FP / FN / TN |
|---|---|---|---|---|---|---|
| test | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 93 / 0 / 0 / 0 |
| val | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 | 80 / 0 / 0 / 0 |
| train | 1.000 | 0.990 | 0.995 | 0.000 | 0.990 | 406 / 0 / 4 / 0 |
| all | 1.000 | 0.993 | 0.997 | 0.000 | 0.993 | 579 / 0 / 4 / 0 |

## Sâu răng — toàn bộ ảnh

Coi ảnh không có polygon sâu răng là không sâu (bi quan cho precision nếu bác sĩ bỏ sót).

**Mức răng** — đơn vị: răng app tìm thấy; nhãn = 1 nếu có vùng sâu bác sĩ giao nhiều nhất với răng đó; dự đoán = 1 nếu app báo ≥ 1 vùng sâu.

| Split | Precision | Recall | F1 | Specificity | Accuracy | TP / FP / FN / TN |
|---|---|---|---|---|---|---|
| test | 0.406 | 0.807 | 0.541 | 0.913 | 0.905 | 213 / 311 / 51 / 3246 |
| val | 0.427 | 0.870 | 0.573 | 0.917 | 0.914 | 220 / 295 / 33 / 3253 |
| train | 0.454 | 0.902 | 0.604 | 0.921 | 0.920 | 1077 / 1297 / 117 / 15098 |
| all | 0.442 | 0.882 | 0.589 | 0.919 | 0.916 | 1510 / 1903 / 201 / 21597 |

**Mức vùng** — ghép 1-1 tham lam theo IoU polygon ≥ 0.2.

| Split | Ảnh | Vùng GT | Vùng dự đoán | Precision | Recall | F1 | GT không thuộc răng nào |
|---|---|---|---|---|---|---|---|
| test | 134 | 295 | 557 | 0.377 | 0.712 | 0.493 | 4 |
| val | 134 | 282 | 538 | 0.409 | 0.780 | 0.537 | 13 |
| train | 623 | 1327 | 2518 | 0.438 | 0.830 | 0.573 | 47 |
| all | 891 | 1904 | 3613 | 0.424 | 0.805 | 0.555 | 64 |

**Mức ảnh** — có / không có sâu răng.

| Split | Precision | Recall | F1 | Specificity | Accuracy | TP / FP / FN / TN |
|---|---|---|---|---|---|---|
| test | 0.756 | 1.000 | 0.861 | 0.268 | 0.776 | 93 / 30 / 0 / 11 |
| val | 0.640 | 1.000 | 0.780 | 0.167 | 0.664 | 80 / 45 / 0 / 9 |
| train | 0.702 | 0.990 | 0.822 | 0.193 | 0.718 | 406 / 172 / 4 / 41 |
| all | 0.701 | 0.993 | 0.822 | 0.198 | 0.718 | 579 / 247 / 4 / 61 |

## Ảnh combine

`vis/<split>/<doc>_<ảnh>.jpg` — trái: nhãn bác sĩ (viền răng xám, vùng sâu đỏ, box răng khôn lệch ngầm đỏ); phải: dự đoán như webapp (polygon răng màu theo cung hàm + FDI, vùng sâu viền đỏ + conf, ✗ = không khớp nhãn; trục răng khôn đỏ = ngầm/lệch, xanh lá = bình thường, trục răng 7 xanh dương đứt nét, mặt nhai vàng, CEJ tím).
