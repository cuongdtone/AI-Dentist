"""Tạo trang HTML duyệt nhanh phân loại Winter (không cần bác sĩ: Winter chỉ là góc trục răng).
Dùng: python make_review.py --features features_test.csv --n 120 --out review
Mở review/index.html bằng trình duyệt -> bấm Đúng / chọn lớp đúng -> "Tải kết quả" (review_result.json).
Rồi chạy: python make_review.py --score review_result.json   để xem độ chính xác Winter.
"""
import argparse, csv, json, os, random, sys
from collections import Counter
import cv2
from ultralytics import YOLO
from wisdom_geom import WINTER_VI, analyze_tooth, draw, load_params, predict_masks

sys.stdout.reconfigure(encoding="utf-8")
CLASSES = ["vertical", "mesioangular", "horizontal", "distoangular", "other"]

HTML = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Duyệt Winter</title><style>
body{font-family:system-ui,sans-serif;margin:0;background:#111;color:#eee}
header{position:sticky;top:0;background:#222;padding:10px 16px;display:flex;gap:16px;align-items:center;z-index:2}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(340px,1fr));gap:12px;padding:16px}
.card{background:#1c1c1c;border:2px solid #333;border-radius:8px;padding:8px}.card.ok{border-color:#2a7}.card.fix{border-color:#d84}
.card img{width:100%;border-radius:4px}.meta{font-size:14px;margin:6px 0}.btns button{margin:2px;padding:4px 8px;cursor:pointer;background:#333;color:#eee;border:1px solid #555;border-radius:4px}
.btns button.sel{background:#2a7}button.dl{padding:6px 12px}
</style></head><body><header><b>Duyệt phân loại Winter</b><span id="prog"></span>
<button class="dl" onclick="dl()">Tải kết quả</button><span style="font-size:13px;opacity:.7">Trục đỏ/xanh lá = răng khôn, trục xanh dương = răng 7. Bấm lớp đúng (mặc định coi lớp dự đoán là đúng khi bấm "Đúng").</span></header>
<div class="grid" id="g"></div><script>
const ITEMS=__ITEMS__, CL=__CL__, VI=__VI__; const KEY='winter_review_v1';
let ans={}; try{ans=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function save(){try{localStorage.setItem(KEY,JSON.stringify(ans))}catch(e){};prog()}
function prog(){document.getElementById('prog').textContent=Object.keys(ans).length+'/'+ITEMS.length+' đã duyệt'}
function render(){const g=document.getElementById('g');g.innerHTML='';ITEMS.forEach(it=>{const a=ans[it.id];
 const d=document.createElement('div');d.className='card'+(a?(a===it.pred?' ok':' fix'):'');
 d.innerHTML=`<img loading="lazy" src="${it.img}"><div class="meta">${it.id} — dự đoán: <b>${VI[it.pred]}</b> ${it.angle>0?'+':''}${it.angle}°</div>
 <div class="btns"><button onclick="setA('${it.id}','${it.pred}')">✓ Đúng</button>${CL.map(c=>`<button class="${a===c?'sel':''}" onclick="setA('${it.id}','${c}')">${VI[c]}</button>`).join('')}
 <button onclick="setA('${it.id}','skip')">Bỏ qua</button></div>`;g.appendChild(d)})}
function setA(id,c){ans[id]=c;save();render()}
function dl(){const out=ITEMS.map(it=>({id:it.id,pred:it.pred,angle:it.angle,truth:ans[it.id]||null}));
 const b=new Blob([JSON.stringify(out,null,1)],{type:'application/json'});const u=URL.createObjectURL(b);
 const l=document.createElement('a');l.href=u;l.download='review_result.json';l.click()}
render();prog();
</script></body></html>"""


def score(path):
    rs = [r for r in json.load(open(path)) if r["truth"] and r["truth"] != "skip"]
    ok = sum(r["truth"] == r["pred"] for r in rs)
    print(f"Winter: {ok}/{len(rs)} đúng ({ok / max(len(rs), 1):.1%})")
    print("Nhầm (dự đoán -> đúng):", Counter((r["pred"], r["truth"]) for r in rs if r["truth"] != r["pred"]).most_common())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_test.csv")
    ap.add_argument("--src", default="../dentex/extracted/disease/input")
    ap.add_argument("--weights", default="../tooth-detect/runs/segment/runs/yolo26s-seg_1024/weights/best.pt")
    ap.add_argument("--params", default="params.json")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--out", default="review")
    ap.add_argument("--device", default=None)
    ap.add_argument("--score", default=None, help="file review_result.json để chấm điểm")
    a = ap.parse_args()
    if a.score:
        return score(a.score)

    rows = list(csv.DictReader(open(a.features)))
    # lấy mẫu cân bằng theo lớp Winter dự đoán để các lớp hiếm cũng được duyệt
    by = {}
    for r in rows:
        by.setdefault(r["winter"], []).append(r)
    rng = random.Random(0); pick = []
    for k in by:
        rng.shuffle(by[k])
    while len(pick) < a.n and any(by.values()):
        for k in list(by):
            if by[k] and len(pick) < a.n:
                pick.append(by[k].pop())

    p = load_params(a.params); model = YOLO(a.weights)
    os.makedirs(f"{a.out}/img", exist_ok=True)
    items, cache = [], {}
    for r in sorted(pick, key=lambda r: r["image"]):
        if r["image"] not in cache:
            img = cv2.imread(os.path.join(a.src, r["image"]))
            cache = {r["image"]: (img, predict_masks(model, img, device=a.device))}
        img, masks = cache[r["image"]]
        f = int(r["fdi"])
        if f not in masks:
            continue
        res = analyze_tooth(masks[f], masks.get(f - 1), f, p)
        if "winter" not in res:
            continue
        vis = draw(img, [res], {f: masks[f]})
        cx, cy = map(int, res["m3_center"]); H, W = img.shape[:2]; s = 420
        crop = vis[max(cy - s, 0):cy + s, max(cx - int(s * 1.3), 0):cx + int(s * 1.3)]
        iid = f"{r['image'][:-4]}_{f}"
        cv2.imwrite(f"{a.out}/img/{iid}.jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
        items.append(dict(id=iid, img=f"img/{iid}.jpg", pred=res["winter"], angle=res["winter_angle"]))
    html = (HTML.replace("__ITEMS__", json.dumps(items)).replace("__CL__", json.dumps(CLASSES))
            .replace("__VI__", json.dumps(WINTER_VI, ensure_ascii=False)))
    open(f"{a.out}/index.html", "w", encoding="utf8").write(html)
    print(f"{len(items)} răng -> {a.out}/index.html  ({Counter(i['pred'] for i in items)})")


if __name__ == "__main__":
    main()
