# perio-seg — mức CEJ + mức xương nha chu

Nhãn final_data là **vùng khép kín theo cả cung hàm** (không theo từng răng):

| Nhãn | Vòng / ảnh | Lớp | Ý nghĩa |
|---|---|---|---|
| `Muc CEJ` | 2 | `crown_upper`, `crown_lower` | vùng thân răng; mép phía chân = đường CEJ |
| `Muc xuong nha chu` | 1 | `bone_window` | vùng giữa đỉnh xương ổ răng 2 hàm; mép = đường đỉnh xương |

Model YOLO26-seg 3 lớp trên ảnh nguyên khung (1024) → đo từng răng bằng hình học
([app/backend/app/perio_geom.py](../../app/backend/app/perio_geom.py)): điểm CEJ + đỉnh xương ở mặt gần / xa,
% tiêu xương = (CEJ→đỉnh xương − khoảng sinh lý) / (CEJ→chóp), giai đoạn AAP/EFP 2017 (<15% I · 15–33% II · >33% III/IV).

```bash
P=../../wisdom_tooth/env/bin/python
$P prepare_data.py                    # -> data/ (713 ảnh: train 501 / val 107 / test 105), data/vis/ để kiểm tra
$P train.py --epochs 100 --batch 4 --workers 0     # -> runs/segment/runs/perio_1024
cd ../../app/scripts && $P export_onnx.py --preset perio   # -> app/backend/models/perio_seg.onnx
cd ../../research/e2e-evaluation && $P evaluate.py         # tự lấy best.pt mới nhất, có phần CEJ / tiêu xương
```

Bỏ khi tạo dataset: ảnh không có nhãn CEJ/xương (169, phần lớn ảnh trẻ em), ảnh vẽ sai số vòng (9), trùng hash (1).
Lưu ý: nhãn vẽ thô (~50 điểm cả cung hàm) → đủ cho sàng lọc, không phải đo mm lâm sàng.
