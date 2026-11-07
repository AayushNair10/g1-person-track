"""Python reference of the robot's face + hand pipeline (src/*.hpp), for the laptop: the desktop checks compare
the C++ code against it, and it runs the whole pipeline on a video file with ONNX Runtime to tune gesture rules.

    ~/Downloads/benchmark_face/.venv/bin/python scripts/facehand_ref.py VIDEO [--out annotated.mp4] [--every N]
        [--size 1280x720] [--mirrored]

Prints gesture events like the robot app ("[  3.20 s] right hand: Thumbs up") and, with --out, writes an
annotated video. The followed face here is simply the largest face (the robot uses OC-SORT + the target rules).
"""
import argparse
import math
import os
import sys
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PALM_IN, HAND_IN = 192, 224


# ----------------------------------------------------------------- input preparation (preprocess_math.hpp)

def make_letterbox(x0, y0, w, h, dw, dh):
    r = min(dh / h, dw / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    left, top = int(round((dw - nw) / 2 - 0.1)), int(round((dh - nh) / 2 - 0.1))
    return dict(x0=x0, y0=y0, w=w, h=h, dw=dw, dh=dh, nw=nw, nh=nh, left=left, top=top,
                gx=np.float32(nw / w), gy=np.float32(nh / h))


def lb_to_frame(lb, x, y):
    return (x - lb["left"]) / lb["gx"] + lb["x0"], (y - lb["top"]) / lb["gy"] + lb["y0"]


def letterbox(frame_bgr, lb, nhwc, rgb, scale, pad=0.0):
    """cv2.resize INTER_LINEAR_EXACT of the region (the C++ equals it exactly at 1/2 and 1/4 scale)."""
    import cv2
    reg = frame_bgr[lb["y0"]:lb["y0"] + lb["h"], lb["x0"]:lb["x0"] + lb["w"]]
    img = cv2.resize(reg, (lb["nw"], lb["nh"]), interpolation=cv2.INTER_LINEAR_EXACT).astype(np.float32)
    out = np.full((lb["dh"], lb["dw"], 3), pad / scale if scale else 0, np.float32)
    out[lb["top"]:lb["top"] + lb["nh"], lb["left"]:lb["left"] + lb["nw"]] = img
    if rgb:
        out = out[..., ::-1]
    out = out * np.float32(scale)
    return (out if nhwc else out.transpose(2, 0, 1))[None].copy()


def rot_to_frame(r, dx, dy):
    c, s = math.cos(r[3]), math.sin(r[3])
    return r[0] + dx * c - dy * s, r[1] + dx * s + dy * c


def frame_to_rot(r, x, y):
    c, s = math.cos(r[3]), math.sin(r[3])
    ux, uy = x - r[0], y - r[1]
    return ux * c + uy * s, -ux * s + uy * c


def crop_to_frame(r, out, u, v):
    return rot_to_frame(r, (u / out - 0.5) * r[2], (v / out - 0.5) * r[2])


def rotcrop(frame_bgr, r, out):
    """Rotated square r = (cx, cy, size, angle) -> out x out, RGB 0..1, NHWC; n x n samples when shrinking."""
    h, w = frame_bgr.shape[:2]
    step = r[2] / out
    n = int(min(4, max(1, math.ceil(step - 0.01))))
    img = frame_bgr.astype(np.float64)
    acc = np.zeros((out, out, 3))
    uu, vv = np.meshgrid(np.arange(out), np.arange(out))
    c, s = math.cos(r[3]), math.sin(r[3])
    for j in range(n):
        for i in range(n):
            dx = ((uu + (i + 0.5) / n) / out - 0.5) * r[2]
            dy = ((vv + (j + 0.5) / n) / out - 0.5) * r[2]
            x, y = r[0] + dx * c - dy * s, r[1] + dx * s + dy * c
            fx, fy = x - 0.5, y - 0.5
            inside = (fx >= -0.5) & (fy >= -0.5) & (fx <= w - 0.5) & (fy <= h - 0.5)
            x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
            ax, ay = (fx - x0)[..., None], (fy - y0)[..., None]
            x1, y1 = np.clip(x0 + 1, 0, w - 1), np.clip(y0 + 1, 0, h - 1)
            x0, y0 = np.clip(x0, 0, w - 1), np.clip(y0, 0, h - 1)
            v = (1 - ay) * ((1 - ax) * img[y0, x0] + ax * img[y0, x1]) + ay * ((1 - ax) * img[y1, x0] + ax * img[y1, x1])
            acc += np.where(inside[..., None], v, 0)
    acc /= n * n
    return (acc[..., ::-1] / 255.0).astype(np.float32)[None]


# ----------------------------------------------------------------- decoding (face_detector.hpp, hands.hpp)

def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def decode_faces(out, lb, fw, fh, conf=0.6, nms=0.3):
    """out: (100, 15) -> list of (x1 y1 x2 y2 score, 10 points) in frame pixels."""
    faces, kept = [], []
    for r in out:
        if not r[4] >= conf:
            break
        if any(iou(r[:4], k) > nms for k in kept):
            continue
        kept.append(r[:4])
        x1, y1 = lb_to_frame(lb, r[0], r[1])
        x2, y2 = lb_to_frame(lb, r[2], r[3])
        pts = [c for p in range(5) for c in lb_to_frame(lb, r[5 + 2 * p], r[6 + 2 * p])]
        faces.append(np.array([min(max(x1, 0), fw), min(max(y1, 0), fh), min(max(x2, 0), fw), min(max(y2, 0), fh),
                               r[4]] + pts, np.float64))
    return faces


def decode_palms(out, lb, conf=0.5, nms=0.3):
    """out: (32, 21) -> list of (x1 y1 x2 y2 score, 14 points) in frame pixels."""
    palms = []
    for r in out:
        score = 1 / (1 + math.exp(-float(r[0])))
        if not score >= conf:
            break
        g = r[1:19]
        ax, ay = r[19] * PALM_IN, r[20] * PALM_IN
        cx, cy, w, h = ax + g[0], ay + g[1], g[2], g[3]
        x1, y1 = lb_to_frame(lb, cx - w / 2, cy - h / 2)
        x2, y2 = lb_to_frame(lb, cx + w / 2, cy + h / 2)
        pts = [c for k in range(7) for c in lb_to_frame(lb, ax + g[4 + 2 * k], ay + g[5 + 2 * k])]
        p = np.array([x1, y1, x2, y2, score] + pts, np.float64)
        if all(iou(p[:4], q[:4]) <= nms for q in palms):
            palms.append(p)
    return palms


def up_angle(x0, y0, x1, y1):
    return math.atan2(x1 - x0, -(y1 - y0))


def roi_from_palm(p):
    pts = p[5:]
    a = up_angle(pts[0], pts[1], pts[4], pts[5])
    w, h = p[2] - p[0], p[3] - p[1]
    r = [(p[0] + p[2]) / 2, (p[1] + p[3]) / 2, 2.6 * max(w, h), a]
    r[0], r[1] = rot_to_frame(r, 0.0, -0.5 * h)
    return tuple(r)


def decode_hand(out, r, size=HAND_IN):
    lm = np.array([crop_to_frame(r, size, out[3 * i], out[3 * i + 1]) for i in range(21)])
    return dict(xy=lm, z=out[2:63:3] * r[2] / size, world=out[65:128].reshape(21, 3).astype(np.float64),
                presence=float(out[63]), handedness=float(out[64]))


def roi_from_landmarks(h):
    xy = h["xy"]
    r = (xy[0, 0], xy[0, 1], 1.0, up_angle(xy[0, 0], xy[0, 1], xy[9, 0], xy[9, 1]))
    d = np.array([frame_to_rot(r, x, y) for x, y in xy])
    lo, hi = d.min(0), d.max(0)
    w, ht = hi[0] - lo[0], hi[1] - lo[1]
    cx, cy = rot_to_frame(r, (lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2 - 0.1 * ht)
    return (cx, cy, 2.0 * max(w, ht), r[3])


def search_region(face, fw, fh, width=6.0, above=1.0, below=3.5):
    w_, h_, cx = face[2] - face[0], face[3] - face[1], (face[0] + face[2]) / 2
    rx1, rx2 = max(0.0, cx - width * w_ / 2), min(float(fw), cx + width * w_ / 2)
    ry1, ry2 = max(0.0, face[1] - above * h_), min(float(fh), face[3] + below * h_)
    x0, y0 = int(rx1), int(ry1)
    return x0, y0, max(1, int(rx2) - x0), max(1, int(ry2) - y0)


# ----------------------------------------------------------------- gestures (gestures.cpp)

CFG = dict(straight_bend_deg=50.0, curled_bend_deg=100.0, thumb_straight_deg=45.0, thumb_out=0.75, touch=0.30,
           dir_ratio=0.6)


def _bend(a, b, c):
    u, v = b - a, c - b
    d = np.linalg.norm(u) * np.linalg.norm(v)
    if d <= 0:
        return 0.0
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(u, v) / d)))))


def hand_shape(h, c=CFG):
    W = h["world"]
    dist = lambda i, j: float(np.linalg.norm(W[i] - W[j]))  # noqa: E731
    palm = max(dist(0, 9), 1e-4)
    f = [None] * 5
    for k in range(1, 5):
        b = 1 + 4 * k
        p, d, t = b + 1, b + 2, b + 3
        bent = _bend(W[b], W[p], W[d]) + _bend(W[p], W[d], W[t])
        tip, mid, base = dist(t, 0), dist(p, 0), dist(b, 0)
        if bent < c["straight_bend_deg"] and tip > mid:
            f[k] = 2
        elif bent > c["curled_bend_deg"] or tip < base * 1.05:
            f[k] = 0
        else:
            f[k] = 1
    tb = _bend(W[1], W[2], W[3]) + _bend(W[2], W[3], W[4])
    out = dist(4, 5) / palm
    f[0] = 2 if (tb < c["thumb_straight_deg"] and out > c["thumb_out"]) else (0 if out < 0.5 * c["thumb_out"] else 1)
    return f, dist(4, 8) / palm


def _direction(dx, dy, ratio):
    ax, ay, n = abs(dx), abs(dy), math.hypot(dx, dy)
    if n <= 0:
        return None
    if ay >= ax and ay >= ratio * n:
        return "up" if dy < 0 else "down"
    if ax > ay and ax >= ratio * n:
        return "left" if dx < 0 else "right"
    return None


def classify(h, c=CFG):
    f, touch = hand_shape(h, c)
    T, I, M, R, L = (v == 2 for v in f)
    Tc = f[0] != 2
    Ic, Mc, Rc, Lc = (v == 0 for v in f[1:])
    xy = h["xy"]
    if touch < c["touch"]:
        if M and R and L:
            return "OK"
        if not (M and R):
            return "Pinch"
    if T and I and M and R and L:
        return "Open palm"
    if Ic and Mc and Rc and Lc:
        if T:
            d = _direction(xy[4, 0] - xy[2, 0], xy[4, 1] - xy[2, 1], c["dir_ratio"])
            return {"up": "Thumbs up", "down": "Thumbs down"}.get(d, "")
        return "Fist"
    if T and Ic and Mc and Rc and L:
        return "Call me"
    if I and Mc and Rc and L:
        return "I love you" if T else "Rock"
    if I and M and R and Lc and Tc:
        return "Three"
    if I and M and Rc and Lc:
        return "Victory"
    if I and Mc and Rc and Lc:
        d = _direction(xy[8, 0] - xy[5, 0], xy[8, 1] - xy[5, 1], c["dir_ratio"])
        if d:
            return "Pointing " + d
    return ""


class Smoother:
    def __init__(self, window=7, need=4):
        self.hist, self.window, self.need, self.stable = deque(), window, need, ""

    def update(self, label):
        self.hist.append(label)
        while len(self.hist) > self.window:
            self.hist.popleft()
        best, n = max(((a, sum(b == a for b in self.hist)) for a in self.hist), key=lambda t: t[1])
        self.stable = best if n >= self.need else ""
        return self.stable


def person_side(handedness, mirrored=False):
    right = handedness > 0.5
    return "right" if (right if mirrored else not right) else "left"


# ----------------------------------------------------------------- laptop runner (ONNX Runtime)

class Pipeline:
    def __init__(self, models=ROOT / "models", face="yunet_640x384.onnx", mirrored=False):
        import onnxruntime as ort
        opt = ["CPUExecutionProvider"]
        self.face = ort.InferenceSession(str(models / face), providers=opt)
        self.palm = ort.InferenceSession(str(models / "palm_192.onnx"), providers=opt)
        self.hand = ort.InferenceSession(str(models / "handpose_224.onnx"), providers=opt)
        shp = self.face.get_inputs()[0].shape
        self.fin = (shp[3], shp[2])
        self.hands, self.frame, self.mirrored = [], 0, mirrored

    def step(self, f):
        H, W = f.shape[:2]
        lb = make_letterbox(0, 0, W, H, *self.fin)
        out = self.face.run(None, {"input": letterbox(f, lb, False, False, 1.0)})[0][0]
        faces = decode_faces(out, lb, W, H)
        events = []
        if not faces:
            self.hands = []
        else:
            tf = max(faces, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
            fw, fcx, fcy = tf[2] - tf[0], (tf[0] + tf[2]) / 2, (tf[1] + tf[3]) / 2
            kept = []
            for h in self.hands:
                o = self.hand.run(None, {"input_1": rotcrop(f, h["roi"], HAND_IN)})[0][0]
                lm = decode_hand(o, h["roi"])
                if lm["presence"] < 0.5 or math.hypot(lm["xy"][9, 0] - fcx, lm["xy"][9, 1] - fcy) > 6 * fw:
                    continue
                h["lm"] = lm
                kept.append(h)
            if len(kept) == 2 and math.hypot(kept[0]["roi"][0] - kept[1]["roi"][0], kept[0]["roi"][1] - kept[1]["roi"][1]) \
                    < 0.5 * min(kept[0]["roi"][2], kept[1]["roi"][2]):
                kept.pop(1 if kept[0]["lm"]["presence"] >= kept[1]["lm"]["presence"] else 0)
            self.hands = kept
            if len(self.hands) < 2 and (not self.hands or self.frame % 5 == 0):
                x0, y0, rw, rh = search_region(tf, W, H)
                lbp = make_letterbox(x0, y0, rw, rh, PALM_IN, PALM_IN)
                po = self.palm.run(None, {"input_1": letterbox(f, lbp, True, True, 1 / 255)})[0][0]
                for p in decode_palms(po, lbp):
                    if len(self.hands) >= 2:
                        break
                    pc = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
                    if any(math.hypot(pc[0] - h["roi"][0], pc[1] - h["roi"][1]) < 0.5 * h["roi"][2] for h in self.hands):
                        continue
                    roi = roi_from_palm(p)
                    lm = decode_hand(self.hand.run(None, {"input_1": rotcrop(f, roi, HAND_IN)})[0][0], roi)
                    if lm["presence"] >= 0.7:
                        self.hands.append(dict(roi=roi, lm=lm, side=lm["handedness"], age=0, sm=Smoother(), rep=""))
            for h in self.hands:
                h["side"] = h["lm"]["handedness"] if h["age"] == 0 else 0.8 * h["side"] + 0.2 * h["lm"]["handedness"]
                h["age"] += 1
                h["raw"] = classify(h["lm"])
                st = h["sm"].update(h["raw"])
                if st != h["rep"]:
                    h["rep"] = st
                    if st:
                        events.append((person_side(h["side"], self.mirrored), st))
                h["roi"] = roi_from_landmarks(h["lm"])
        self.frame += 1
        return faces, self.hands, events


def main():
    import cv2
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--out")
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--size", default="1280x720", help="frames resized to this (the robot's camera mode)")
    ap.add_argument("--mirrored", action="store_true")
    ap.add_argument("--face", default="yunet_640x384.onnx")
    a = ap.parse_args()
    W, H = map(int, a.size.split("x"))
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    pipe = Pipeline(face=a.face, mirrored=a.mirrored)
    writer = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps / a.every, (W, H)) if a.out else None
    i = 0
    while True:
        ok, f = cap.read()
        if not ok:
            break
        i += 1
        if (i - 1) % a.every:
            continue
        f = cv2.resize(f, (W, H), interpolation=cv2.INTER_AREA)
        faces, hands, events = pipe.step(f)
        for side, g in events:
            print("[%7.2f s] %s hand: %s" % (i / fps, side, g))
        if writer:
            for fc in faces:
                cv2.rectangle(f, (int(fc[0]), int(fc[1])), (int(fc[2]), int(fc[3])), (0, 220, 0), 2)
            for h in hands:
                for x, y in h["lm"]["xy"]:
                    cv2.circle(f, (int(x), int(y)), 3, (0, 160, 255), -1)
                cv2.putText(f, "%s: %s (raw %s)" % (person_side(h["side"], a.mirrored), h["rep"] or "-", h["raw"] or "-"),
                            (int(h["lm"]["xy"][:, 0].min()), max(20, int(h["lm"]["xy"][:, 1].min()) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 160, 255), 2)
            writer.write(f)
    if writer:
        writer.release()
        print("wrote", a.out)


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    sys.exit(main())
