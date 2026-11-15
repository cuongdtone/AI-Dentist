import json, os, sqlite3, threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

DATA_DIR = os.getenv("DATA_DIR", "./data")
TZ = ZoneInfo(os.getenv("TZ", "Asia/Ho_Chi_Minh"))
COUNTS = ["teeth", "wisdom", "caries", "perio", "cej"]  # các chỉ số thống kê

os.makedirs(f"{DATA_DIR}/images", exist_ok=True)
_conn = sqlite3.connect(f"{DATA_DIR}/app.db", check_same_thread=False)
_conn.row_factory = sqlite3.Row
_lock = threading.Lock()
_conn.executescript(f"""
CREATE TABLE IF NOT EXISTS users (username TEXT PRIMARY KEY, salt TEXT, pw_hash TEXT);
CREATE TABLE IF NOT EXISTS inferences (
  id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT, username TEXT, filename TEXT, image_path TEXT,
  width INT, height INT, {", ".join(f"{c} INT DEFAULT 0" for c in COUNTS)}, result TEXT);
CREATE INDEX IF NOT EXISTS idx_inf_time ON inferences(created_at);
""")


def q(sql, args=(), one=False):
    with _lock:
        cur = _conn.execute(sql, args)
        _conn.commit()
        rows = cur.fetchall()
        return (rows[0] if rows else None) if one else rows


def now():
    return datetime.now(TZ)


def save_inference(username, filename, image_path, result, counts):
    with _lock:
        cur = _conn.execute(
            f"INSERT INTO inferences (created_at, username, filename, image_path, width, height, {', '.join(COUNTS)}, result)"
            f" VALUES (?,?,?,?,?,?,{','.join('?' * len(COUNTS))},?)",
            (now().strftime("%Y-%m-%d %H:%M:%S"), username, filename, image_path, result["width"], result["height"],
             *[counts.get(c, 0) for c in COUNTS], json.dumps(result)))
        _conn.commit()
        return cur.lastrowid


# period -> (số mốc hiển thị, số ngày nhìn lại để phủ đủ các mốc, hàm tạo khoá nhóm)
PERIODS = {
    "day": (30, 30, lambda d: d.strftime("%Y-%m-%d")),
    "week": (12, 7 * 12, lambda d: d.strftime("%G-W%V")),
    "month": (12, 366, lambda d: d.strftime("%Y-%m")),
    "year": (5, 366 * 5, lambda d: d.strftime("%Y")),
}


def stats(period):
    n, days, key = PERIODS[period]
    today = now().date()
    start = today - timedelta(days=days - 1)
    buckets = {}
    for i in range(days):  # tạo đủ các mốc (kể cả mốc không có dữ liệu)
        buckets.setdefault(key(start + timedelta(days=i)), {"images": 0, **{c: 0 for c in COUNTS}})
    buckets = dict(list(buckets.items())[-n:])
    rows = q(f"SELECT created_at, {', '.join(COUNTS)} FROM inferences WHERE created_at >= ?", (start.isoformat(),))
    for r in rows:
        b = buckets.get(key(datetime.strptime(r["created_at"][:10], "%Y-%m-%d")))
        if b is None:  # nằm ngoài n mốc cuối
            continue
        b["images"] += 1
        for c in COUNTS:
            b[c] += r[c]
    series = [{"bucket": k, **v} for k, v in buckets.items()]
    total = q(f"SELECT COUNT(*) AS images, {', '.join(f'COALESCE(SUM({c}),0) AS {c}' for c in COUNTS)} FROM inferences", one=True)
    in_range = {k: sum(s[k] for s in series) for k in ["images", *COUNTS]}
    return {"period": period, "series": series, "range": in_range, "all_time": dict(total)}
