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
```
Mặc định lấy `best.pt` của `research/tooth-detect/.../ft_finaldata_30` và `research/caries-seg/.../caries_384`; đổi bằng `--weights`.
File cũ được sao lưu thành `*.prev.onnx`. Thiếu `caries_seg.onnx` thì bước Sâu răng báo "Chưa có model sâu răng" (không lỗi).

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
Thống kê đếm `len(findings[k])` với `k` thuộc `wisdom, caries, perio, cej`.
Config của lần chạy được lưu trong `result["config"]`, nên khi lưu chỉnh sửa, phần răng khôn được tính lại với cùng tham số.

## Viewer

- Lăn chuột: zoom quanh con trỏ · kéo: pan · `Fit`
- Mặc định: viền mờ + số FDI. Tuỳ chọn độc lập: Mask, Polygon, FDI, Opacity
- Chế độ *Chỉnh sửa*: kéo đỉnh = di chuyển, kéo cạnh = thêm đỉnh, chuột phải vào đỉnh = xoá đỉnh; bút chì / tẩy; Ctrl+Z / Ctrl+Y → `Lưu`
- Graph: ⚙ = cấu hình bước; chuột phải vào node = Configure / Disable·Enable / Hide·Show lớp vẽ
