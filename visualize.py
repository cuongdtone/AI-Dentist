"""Xem dataset Supervisely (ann/img/meta.json).
Dùng:  python visualize.py final_data/Lan            # xem tương tác
       python visualize.py final_data/Lan --save out  # lưu toàn bộ ảnh đã vẽ
Phím:  d/→ ảnh sau | a/← ảnh trước | f lọc ảnh có lớp 'sâu răng' | space bật/tắt nhãn | s lưu | q thoát
"""
import argparse, json, os, sys
from collections import Counter
import cv2
import numpy as np


def load(root):
    meta = json.load(open(f"{root}/meta.json", encoding="utf8"))
    col = {c["title"]: tuple(int(c["color"][i:i + 2], 16) for i in (5, 3, 1)) for c in meta["classes"]}  # BGR
    names = sorted(f[:-5] for f in os.listdir(f"{root}/ann"))
    return col, names


def render(root, name, col, show=True, height=900):
    img = cv2.imdecode(np.fromfile(f"{root}/img/{name}", np.uint8), cv2.IMREAD_COLOR)  # hỗ trợ đường dẫn unicode
    ann = json.load(open(f"{root}/ann/{name}.json", encoding="utf8"))
    cnt = Counter(o["classTitle"] for o in ann["objects"])
    if show:
        ov = img.copy()
        for o in ann["objects"]:
            c = col.get(o["classTitle"], (255, 255, 255))
            p = np.array(o["points"]["exterior"], np.int32)
            g = o["geometryType"]
            if g == "polygon":
                cv2.fillPoly(ov, [p], c)
                cv2.polylines(img, [p], True, c, 2)
            elif g == "rectangle":
                cv2.rectangle(img, tuple(p[0]), tuple(p[1]), c, 2)
            else:  # line
                cv2.polylines(img, [p], False, c, 2)
        img = cv2.addWeighted(ov, 0.3, img, 0.7, 0)
    s = height / img.shape[0]
    img = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    cv2.putText(img, f"{name}  {ann['size']['width']}x{ann['size']['height']}", (10, 25), 0, 0.7, (255, 255, 255), 2)
    for i, (k, v) in enumerate(sorted(cnt.items())):  # chú thích + số lượng
        cv2.putText(img, f"{k}: {v}", (10, 55 + 25 * i), 0, 0.6, col.get(k, (255, 255, 255)), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--save")
    a = ap.parse_args()
    col, names = load(a.root)
    print(f"{len(names)} ảnh | lớp: {list(col)}")
    if a.save:
        os.makedirs(a.save, exist_ok=True)
        for n in names:
            cv2.imencode(".jpg", render(a.root, n, col))[1].tofile(f"{a.save}/{n}.jpg")
        return print("đã lưu vào", a.save)
    i, show, only = 0, True, False
    while True:
        print(f"{i + 1}/{len(names)}: {names[i]}")
        cv2.imshow("viz", render(a.root, names[i], col, show))
        k = cv2.waitKeyEx(0)
        step = 1 if k in (ord("d"), 2555904) else -1 if k in (ord("a"), 2424832) else 0
        if k == ord("q") or k == 27: break
        if k == ord(" "): show = not show
        if k == ord("f"): only = not only
        if k == ord("s"):
            cv2.imencode(".jpg", render(a.root, names[i], col, show))[1].tofile(f"{names[i]}.jpg")
        for _ in range(len(names)):  # nhảy tới ảnh kế (có lọc nếu bật f)
            i = (i + step) % len(names)
            if not (only and step) or any("Sau rang" in o["classTitle"] for o in
                    json.load(open(f"{a.root}/ann/{names[i]}.json", encoding="utf8"))["objects"]):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
