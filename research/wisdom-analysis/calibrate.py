"""Chỉnh ngưỡng luật "ngầm/lệch" trên features_train.csv, đánh giá trên features_test.csv.
Dùng: python calibrate.py --train features_train.csv --test features_test.csv --out params.json

Luật: ngầm  <=>  depth_ratio > impact_depth  HOẶC  |góc Winter| > angle_thr
(depth_ratio = (mặt nhai M2 - đỉnh M3) / chiều dài M2; > 0 nghĩa là M3 thấp hơn mặt nhai M2).
Ngưỡng chọn bằng grid search tối đa F1 trên train; test chỉ dùng để báo cáo.
"""
import argparse, csv, json, os, sys
from collections import Counter
import numpy as np
from wisdom_geom import DEFAULT_PARAMS

sys.stdout.reconfigure(encoding="utf-8")


def load(p):
    rows = list(csv.DictReader(open(p)))
    y = np.array([int(r["label"]) for r in rows])
    ang = np.abs(np.array([float(r["angle"]) for r in rows]))
    dep = np.array([float(r["depth_ratio"]) if r["depth_ratio"] != "" else np.nan for r in rows])
    lower = np.array([int(r["fdi"]) // 10 in (3, 4) for r in rows])
    return rows, y, ang, dep, lower


def predict(ang, dep, da, at):
    return ((np.nan_to_num(dep, nan=-9) > da) | (ang > at)).astype(int)


def prf(y, yhat):
    tp = int(((y == 1) & (yhat == 1)).sum()); fp = int(((y == 0) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return dict(P=P, R=R, F1=2 * P * R / max(P + R, 1e-9), Acc=float((y == yhat).mean()), n=len(y))


def fmt(m):
    return f"P={m['P']:.3f} R={m['R']:.3f} F1={m['F1']:.3f} Acc={m['Acc']:.3f} (n={m['n']})"


def summary(name, rows, y):
    print(f"[{name}] {len(rows)} răng khôn, ngầm {y.sum()} ({y.mean():.1%})")
    c = Counter(r["winter"] for r in rows)
    print("  Winter:", ", ".join(f"{k} {v} (ngầm {sum(int(r['label']) for r in rows if r['winter'] == k)})" for k, v in c.most_common()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="features_train.csv")
    ap.add_argument("--test", default="features_test.csv")
    ap.add_argument("--out", default="params.json")
    a = ap.parse_args()

    rows, y, ang, dep, lower = load(a.train)
    summary("train", rows, y)
    grid_d = np.round(np.arange(-0.3, 0.61, 0.01), 2)
    best = max(((prf(y, predict(ang, dep, d, t))["F1"], d, t) for d in grid_d for t in range(10, 181, 2)))
    _, da, at = best
    dep_only = max(((prf(y, predict(ang, dep, d, 999))["F1"], d) for d in grid_d))
    ang_only = max(((prf(y, predict(ang, np.full_like(dep, np.nan), 0, t))["F1"], t) for t in range(5, 181)))
    print(f"  chỉ độ sâu: F1={dep_only[0]:.3f} (impact_depth={dep_only[1]:.2f}) | chỉ góc: F1={ang_only[0]:.3f} (>{ang_only[1]}°)")
    print(f"  LUẬT: depth_ratio > {da:.2f} HOẶC |góc| > {at}°")
    yh = predict(ang, dep, da, at)
    print("  train tất cả :", fmt(prf(y, yh)))
    print("  train hàm dưới:", fmt(prf(y[lower], yh[lower])), "| hàm trên:", fmt(prf(y[~lower], yh[~lower])))

    params = dict(DEFAULT_PARAMS, impact_depth=float(da), angle_thr=float(at))
    json.dump(params, open(a.out, "w"), indent=2)
    print(f"-> lưu {a.out}")

    if a.test and os.path.exists(a.test):
        rows, y, ang, dep, lower = load(a.test)
        print()
        summary("test", rows, y)
        yh = predict(ang, dep, da, at)
        print("  test tất cả  :", fmt(prf(y, yh)))
        print("  test hàm dưới:", fmt(prf(y[lower], yh[lower])), "| hàm trên:", fmt(prf(y[~lower], yh[~lower])))


if __name__ == "__main__":
    main()
