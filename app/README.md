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
/mnt/work/tooth/wisdom_tooth/env/bin/python export_onnx.py   # mặc định lấy best.pt của yolo26s-seg_1024
```

Model được mount vào container (`./backend/models`), nên sau khi export lại chỉ cần `docker compose restart backend`.

## 2. Chạy

```bash
cd app
ADMIN_PASSWORD=... JWT_SECRET=... docker compose up -d --build
# http://localhost:8080  (mặc định admin / admin)
```

Biến môi trường: `PORT` (8080), `ADMIN_USER`, `ADMIN_PASSWORD`, `JWT_SECRET`.
Dữ liệu (SQLite + ảnh upload) nằm trong volume `app-data`.

Dev: `uvicorn app.main:app --reload` trong `backend/` và `npm run dev` trong `frontend/` (Vite proxy `/api` -> :8000).

## Thêm bước pipeline

Trong [backend/app/pipeline.py](backend/app/pipeline.py): viết `def step_x(image, res)` (ghi kết quả vào `res["findings"]["x"]` dạng list),
đăng ký vào `RUNNERS`. Bước có trong `STEPS` mà chưa có trong `RUNNERS` sẽ hiển thị "Chưa có model".
Thống kê đếm `len(findings[k])` với `k` thuộc `wisdom, caries, perio, cej`.

## Viewer

- Lăn chuột: zoom quanh con trỏ · kéo: pan · `Fit` / `1:1`
- Mặc định: viền mờ + số FDI. Tuỳ chọn độc lập: Mask, Polygon, FDI, Opacity
- Bật Polygon để sửa: kéo đỉnh = di chuyển, kéo cạnh = thêm đỉnh, chuột phải vào đỉnh = xoá đỉnh → `Lưu chỉnh sửa`
