"""YOLO26-seg ONNX -> danh sách răng {fdi, conf, box, polygon} theo toạ độ ảnh gốc."""
import ast
import cv2
import numpy as np
import onnxruntime as ort


class ToothSegmenter:
    def __init__(self, path: str, conf: float = 0.25, iou: float = 0.5):
        self.sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.size = self.sess.get_inputs()[0].shape[2]  # ảnh vuông, vd 1024
        self.names = ast.literal_eval(self.sess.get_modelmeta().custom_metadata_map["names"])
        self.conf, self.iou = conf, iou

    def _letterbox(self, img):
        h, w = img.shape[:2]
        r = self.size / max(h, w)
        nw, nh = round(w * r), round(h * r)
        px, py = (self.size - nw) // 2, (self.size - nh) // 2
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        canvas[py:py + nh, px:px + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
        return canvas, r, px, py

    def _decode(self, out, nm, conf, iou):
        """Trả về xyxy, score, cls, coef. Hỗ trợ cả output thô [4+nc+nm, N] (cần NMS) và end2end [N, 6+nm]."""
        nc = len(self.names)
        if out.shape[0] == 4 + nc + nm:  # thô
            p = out.T
            scores_all = p[:, 4:4 + nc]
            cls, scores = scores_all.argmax(1), scores_all.max(1)
            keep = scores > conf
            p, cls, scores = p[keep], cls[keep], scores[keep]
            cx, cy, bw, bh = p[:, :4].T
            xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
            # NMS không phân lớp: mỗi vị trí chỉ là 1 răng
            idx = cv2.dnn.NMSBoxes(np.c_[xyxy[:, :2], p[:, 2:4]].tolist(), scores.tolist(), conf, iou)
            idx = np.array(idx, int).reshape(-1)
            return xyxy[idx], scores[idx], cls[idx], p[idx, 4 + nc:]
        keep = out[:, 4] > conf  # end2end
        p = out[keep]
        return p[:, :4], p[:, 4], p[:, 5].astype(int), p[:, 6:]

    @staticmethod
    def _dedupe(binms, scores, ios_thr=0.6):
        """Loại phát hiện trùng (không phân lớp) theo MASK, duyệt conf giảm dần: bỏ mask có
        giao / diện tích mask nhỏ hơn >= ios_thr với mask đã giữ -> bắt cả mask nhỏ nằm gọn trong mask to
        (IoU thấp nên NMS bỏ sót). Dùng mask chứ không dùng box: box răng nghiêng trùm lên răng bên cạnh."""
        area = [int(m.sum()) for m in binms]
        keep = []
        for i in np.argsort(-np.asarray(scores)):
            if area[i] == 0:
                continue
            if all(np.logical_and(binms[i], binms[j]).sum() / min(area[i], area[j]) < ios_thr for j in keep):
                keep.append(i)
        return sorted(keep)

    def __call__(self, img, conf=None, iou=None, mask_thr=0.5, ios_thr=0.6, eps=1.0):
        """conf/iou: ngưỡng tin cậy & NMS; mask_thr: ngưỡng xác suất mask; ios_thr: ngưỡng loại mask trùng;
        eps: sai số rút gọn polygon (px, ảnh letterbox)."""
        conf = self.conf if conf is None else conf
        iou = self.iou if iou is None else iou
        h, w = img.shape[:2]
        x, r, px, py = self._letterbox(img)
        blob = np.ascontiguousarray(x[:, :, ::-1].transpose(2, 0, 1)[None], np.float32) / 255
        out, proto = self.sess.run(None, {self.input: blob})
        nm, mh, mw = proto.shape[1:]
        boxes, scores, cls, coef = self._decode(out[0], nm, conf, iou)
        if len(boxes) == 0:
            return []
        masks = 1 / (1 + np.exp(-(coef @ proto[0].reshape(nm, -1)))).reshape(-1, mh, mw)

        binms = []
        for b, m in zip(boxes, masks):
            m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
            x1, y1, x2, y2 = np.clip(b, 0, self.size).astype(int)
            binm = np.zeros(m.shape, np.uint8)
            binm[y1:y2, x1:x2] = m[y1:y2, x1:x2] > mask_thr  # chỉ giữ mask trong box
            binms.append(binm)

        teeth = []
        for i in self._dedupe(binms, scores, ios_thr):
            b, s, c, binm = boxes[i], scores[i], cls[i], binms[i]
            cnts, _ = cv2.findContours(binm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            cnt = cv2.approxPolyDP(max(cnts, key=cv2.contourArea), eps, True).reshape(-1, 2)
            if len(cnt) < 3:
                continue
            pts = (cnt - [px, py]) / r
            pts = np.clip(pts, 0, [w, h])
            box = np.clip((b - [px, py, px, py]) / r, 0, [w, h, w, h])
            teeth.append({
                "fdi": self.names[int(c)], "conf": round(float(s), 3),
                "box": [round(float(v), 1) for v in box],
                "polygon": [[round(float(u), 1), round(float(v), 1)] for u, v in pts],
            })
        teeth.sort(key=lambda t: t["fdi"])
        for i, t in enumerate(teeth):
            t["id"] = i
        return teeth
