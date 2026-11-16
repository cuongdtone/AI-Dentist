# Tooth Diagnosis App

FastAPI (onnxruntime CPU) + React (Vite) + nginx, chạy bằng docker compose.

```
app/
├── scripts/export_onnx.py      # .pt -> backend/models/tooth_seg.onnx
├── backend/app/
│   ├── model.py                # tiền/hậu xử lý YOLO26-seg ONNX -> polygon theo FDI
│   ├── pipeline.py             # các bước pipeline, job chạy nền, huỷ
│   ├── db.py                   # SQLite: users, inferences, thống kê
│   ├── auth.py                 # đăng nhập JWT
│   └── main.py                 # REST API
└── frontend/src/
    ├── pages/{Login,Dashboard,Diagnosis}.jsx
    └── components/{PipelineGraph,ImageViewer}.jsx
```

## 1. Export model

```bash
cd app/scripts
/mnt/work/tooth/wisdom_tooth/env/bin/python export_onnx.py --preset tooth    # -> backend/models/tooth_seg.onnx  (1024)
/mnt/work/tooth/wisdom_tooth/env/bin/python export_onnx.py --preset caries   # -> backend/models/caries_seg.onnx (384, crop từng răng)
/mnt/work/tooth/wisdom_tooth/env/bin/python export_onnx.py --preset perio    # -> backend/models/perio_seg.onnx  (1024, vùng CEJ + xương)
```
Mặc định lấy `best.pt` của `research/tooth-detect/.../ft_finaldata_30` và `research/caries-seg/.../caries_384`; đổi bằng `--weights`.
File cũ được sao lưu thành `*.prev.onnx`. Thiếu `caries_seg.onnx` thì bước Sâu răng báo "Chưa có model sâu răng" (không lỗi); thiếu `perio_seg.onnx` thì
bước Mức CEJ báo "Chưa có model CEJ / xương" và bước Tiêu xương nha chu bị bỏ qua.

Model được mount vào container (`./backend/models`), nên sau khi export lại chỉ cần `docker compose restart backend`.

## 2. Chạy

```bash
cd app
ADMIN_PASSWORD=... JWT_SECRET=... docker compose up -d --build
# http://localhost:8112  (mặc định admin / admin)
```

Biến môi trường: `PORT` (8112), `ADMIN_USER`, `ADMIN_PASSWORD`, `JWT_SECRET`.
Dữ liệu (SQLite + ảnh upload) nằm trong volume `app-data`.

Dev: `uvicorn app.main:app --reload` trong `backend/` và `npm run dev` trong `frontend/` (Vite proxy `/api` -> :8000).

## Thêm bước pipeline

Trong [backend/app/pipeline.py](backend/app/pipeline.py):

1. Viết `def step_x(image, res, p)`: ghi kết quả vào `res["findings"]["x"]` dạng list; `p` là config của bước
   (`p["enabled"]` + các tham số).
2. Đăng ký vào `RUNNERS`. Nếu bước cần kết quả của bước khác thì khai báo trong `REQUIRES`.
3. (Tuỳ chọn) Khai báo tham số trong `CONFIG_SCHEMA["x"]`: `type` là `number` (min/max/step) hoặc `select` (options),
   kèm `label`, `default`, `hint`. UI tự sinh hộp cấu hình (nút ⚙ trên node) từ schema này.

Bước có trong `STEPS` nhưng chưa có trong `RUNNERS` hiển thị "Chưa có model".
Thống kê đếm `len(findings[k])` với `k` thuộc `wisdom, caries, cej`; riêng `perio` = số răng tiêu xương giai đoạn ≥ II.
Config của lần chạy được lưu trong `result["config"]`, nên khi lưu chỉnh sửa, CEJ / tiêu xương / răng khôn được tính lại với cùng tham số.

Thứ tự: `seg → cej → wisdom → caries → perio`. Bước `cej` chạy model perio-seg ra 3 vùng theo cả cung hàm
(`result["regions"]`: thân răng trên / dưới, vùng giữa đỉnh xương 2 hàm) và điểm CEJ mặt gần/xa từng răng;
`wisdom` dùng CEJ thật của răng 7 (nếu có) làm ranh giới độ sâu B/C; `perio` đo đỉnh xương, % tiêu xương, giai đoạn
(cách đo: [backend/app/perio_geom.py](backend/app/perio_geom.py)). Vùng sửa được ở chế độ Chỉnh sửa (hover viền vùng).

## Viewer

- Lăn chuột: zoom quanh con trỏ · kéo: pan · `Fit`
- Mặc định: viền mờ + số FDI. Tuỳ chọn độc lập: Mask, Polygon, FDI, Opacity
- Chế độ *Chỉnh sửa*: kéo đỉnh = di chuyển, kéo cạnh = thêm đỉnh, chuột phải vào đỉnh = xoá đỉnh; bút chì / tẩy; Ctrl+Z / Ctrl+Y → `Lưu`
- Graph: ⚙ = cấu hình bước; chuột phải vào node = Configure / Disable·Enable / Hide·Show lớp vẽ
