"""Laptop: prepare the face and hand models for TensorRT 8.5 on the robot.

    ~/Downloads/benchmark_face/.venv/bin/python scripts/export_face_hand.py

Source models (downloaded from OpenCV Zoo into models/ if missing; nothing is installed):
  face_detection_yunet_2023mar.onnx           YuNet face detector (MIT)
  palm_detection_mediapipe_2023feb.onnx       MediaPipe palm detector (Apache-2.0)
  handpose_estimation_mediapipe_2023feb.onnx  MediaPipe hand landmarks (Apache-2.0)

Writes (one input and one output each, fixed shapes, checked against the TRT 8.5 ONNX parser):
  models/yunet_<W>x<H>.onnx  in 1x3xHxW (BGR, 0..255)  out 1x100x15: x1 y1 x2 y2 score, 5 face points (x, y);
                             input pixels, the best 100 by score. The decoding (grid offsets, exp, score =
                             sqrt(cls * obj)) and the top-K run inside TensorRT; NMS stays on the CPU (<= 100 boxes).
  models/palm_192.onnx       in 1x192x192x3 (RGB, 0..1)  out 1x32x21: score logit, 18 regressors (box cx cy w h,
                             7 points), anchor x y; the best 32 by score
  models/handpose_224.onnx   in 1x224x224x3 (RGB, 0..1)  out 1x128: 21 landmarks x y z (input pixels), presence,
                             handedness, 21 world landmarks x y z (metres)
  models/face_hand_report.json

Each new graph is checked with ONNX Runtime against the original model on the same inputs (the numbers must be
the originals' numbers, decoded), and YuNet's decoding against OpenCV's own FaceDetectorYN on real frames.
"""
import argparse
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
import trt85_check  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
SOURCES = {
    "face_detection_yunet_2023mar.onnx": ("face_detection_yunet/", "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "palm_detection_mediapipe_2023feb.onnx": ("palm_detection_mediapipe/", "78ff51c38496b7fc8b8ebdb6cc8c1abb02fa6c38427c6848254cdaba57fcce7c"),
    "handpose_estimation_mediapipe_2023feb.onnx": ("handpose_estimation_mediapipe/", "db0898ae717b76b075d9bf563af315b29562e11f8df5027a1ef07b02bef6d81c"),
}
YUNET_SIZES = [(640, 384), (416, 256), (320, 192)]  # (W, H); multiples of 32, 16:9 pictures inside
YUNET_TOPK = 100
PALM_TOPK = 32
PALM_INPUT = 192


def fetch(models):
    for name, (sub, sha) in SOURCES.items():
        p = models / name
        if not p.exists():
            print("downloading", name)
            urllib.request.urlretrieve(ZOO + sub + name, p)
        got = hashlib.sha256(p.read_bytes()).hexdigest()
        if got != sha:
            sys.exit("%s: unexpected checksum %s" % (name, got))


def const(g, name, arr):
    g.initializer.append(numpy_helper.from_array(np.asarray(arr), name))
    return name


def node(g, op, inputs, outputs, **attrs):
    g.node.append(helper.make_node(op, inputs, outputs, name=outputs[0] + "_node", **attrs))
    return outputs[0]


def set_output(g, name, dims):
    del g.output[:]
    g.output.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, dims))


# ------------------------------------------------------------------ YuNet

def yunet_grids(w, h):
    out = []
    for s in (8, 16, 32):
        cols, rows = w // s, h // s
        gy, gx = np.mgrid[0:rows, 0:cols]  # anchor index = row * cols + col
        out.append((s, gx.reshape(-1).astype(np.float32), gy.reshape(-1).astype(np.float32)))
    return out


def make_yunet(src, dst, w, h):
    m = onnx.load(str(src))
    g = m.graph
    inp = g.input[0].type.tensor_type.shape
    inp.dim[2].dim_value, inp.dim[3].dim_value = h, w
    del g.value_info[:]  # the stored shapes are for 640x640
    zero, one, half = const(g, "c_zero", np.float32(0)), const(g, "c_one", np.float32(1)), const(g, "c_half", np.float32(0.5))
    parts = []
    for s, gx, gy in yunet_grids(w, h):
        n = gx.size
        t = "_s%d" % s
        S = const(g, "stride" + t, np.float32(s))
        GX = const(g, "gx" + t, gx.reshape(1, n, 1))
        GY = const(g, "gy" + t, gy.reshape(1, n, 1))
        GXY = const(g, "gxy" + t, np.repeat(np.stack([gx, gy], 1)[None], 5, axis=0).transpose(1, 0, 2).reshape(1, n, 10))
        c = node(g, "Clip", ["cls_%d" % s, zero, one], ["c" + t])
        o = node(g, "Clip", ["obj_%d" % s, zero, one], ["o" + t])
        score = node(g, "Sqrt", [node(g, "Mul", [c, o], ["co" + t])], ["score" + t])
        node(g, "Split", ["bbox_%d" % s], ["dx" + t, "dy" + t, "dw" + t, "dh" + t], axis=2, split=[1, 1, 1, 1])
        cx = node(g, "Mul", [node(g, "Add", ["dx" + t, GX], ["ax" + t]), S], ["cx" + t])
        cy = node(g, "Mul", [node(g, "Add", ["dy" + t, GY], ["ay" + t]), S], ["cy" + t])
        hw = node(g, "Mul", [node(g, "Mul", [node(g, "Exp", ["dw" + t], ["ew" + t]), S], ["w" + t]), half], ["hw" + t])
        hh = node(g, "Mul", [node(g, "Mul", [node(g, "Exp", ["dh" + t], ["eh" + t]), S], ["h" + t]), half], ["hh" + t])
        x1 = node(g, "Sub", [cx, hw], ["x1" + t])
        y1 = node(g, "Sub", [cy, hh], ["y1" + t])
        x2 = node(g, "Add", [cx, hw], ["x2" + t])
        y2 = node(g, "Add", [cy, hh], ["y2" + t])
        kp = node(g, "Mul", [node(g, "Add", ["kps_%d" % s, GXY], ["kg" + t]), S], ["kp" + t])
        parts.append(node(g, "Concat", [x1, y1, x2, y2, score, kp], ["det" + t], axis=2))
    det = node(g, "Concat", parts, ["det_all"], axis=1)
    det2 = node(g, "Squeeze", [det], ["det_rows"], axes=[0])
    scores = node(g, "Gather", [det2, const(g, "c_col4", np.int64(4))], ["det_scores"], axis=1)
    k = const(g, "c_topk", np.array([YUNET_TOPK], np.int64))
    node(g, "TopK", [scores, k], ["top_scores", "top_idx"], axis=0, largest=1, sorted=1)
    top = node(g, "Gather", [det2, "top_idx"], ["top_rows"], axis=0)
    node(g, "Unsqueeze", [top], ["dets"], axes=[0])
    set_output(g, "dets", [1, YUNET_TOPK, 15])
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def yunet_reference_decode(outs, names, w, h):
    """numpy decode of the ORIGINAL model's 12 outputs (OpenCV FaceDetectorYN's formulas)."""
    o = dict(zip(names, outs))
    rows = []
    for s, gx, gy in yunet_grids(w, h):
        cls = np.clip(o["cls_%d" % s][0, :, 0], 0, 1)
        obj = np.clip(o["obj_%d" % s][0, :, 0], 0, 1)
        b = o["bbox_%d" % s][0]
        k = o["kps_%d" % s][0]
        cx, cy = (b[:, 0] + gx) * s, (b[:, 1] + gy) * s
        bw, bh = np.exp(b[:, 2]) * s, np.exp(b[:, 3]) * s
        kp = (k.reshape(-1, 5, 2) + np.stack([gx, gy], 1)[:, None, :]) * s
        rows.append(np.c_[cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2, np.sqrt(cls * obj), kp.reshape(-1, 10)])
    return np.concatenate(rows)


# ------------------------------------------------------------------ palm detector

def palm_anchors():
    """MediaPipe SSD anchors for the 192x192 palm model (strides 8, 16, 16, 16; 2 anchors per layer and cell,
    fixed size): layers with the same stride share one grid, cell by cell."""
    out = []
    for stride, per_cell in ((8, 2), (16, 6)):
        g = PALM_INPUT // stride
        for y in range(g):
            for x in range(g):
                out += [((x + 0.5) / g, (y + 0.5) / g)] * per_cell
    return np.array(out, np.float32)


def make_palm(src, dst):
    m = onnx.load(str(src))
    g = m.graph
    reg, score = g.output[0].name, g.output[1].name  # (1, 2016, 18), (1, 2016, 1)
    anchors = const(g, "palm_anchors", palm_anchors()[None])
    det = node(g, "Concat", [score, reg, anchors], ["palm_all"], axis=2)
    det2 = node(g, "Squeeze", [det, const(g, "c_ax0", np.array([0], np.int64))], ["palm_rows"])  # opset 13: axes input
    logits = node(g, "Gather", [det2, const(g, "c_col0", np.int64(0))], ["palm_logits"], axis=1)
    node(g, "TopK", [logits, const(g, "c_ptopk", np.array([PALM_TOPK], np.int64))], ["palm_top", "palm_idx"],
         axis=0, largest=1, sorted=1)
    top = node(g, "Gather", [det2, "palm_idx"], ["palm_top_rows"], axis=0)
    node(g, "Unsqueeze", [top, "c_ax0"], ["palms"])
    set_output(g, "palms", [1, PALM_TOPK, 21])
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


def make_handpose(src, dst):
    m = onnx.load(str(src))
    g = m.graph
    names = [o.name for o in g.output]  # landmarks (1,63), presence (1,1), handedness (1,1), world (1,63)
    node(g, "Concat", names, ["hand"], axis=1)
    set_output(g, "hand", [1, 128])
    onnx.checker.check_model(m)
    onnx.save(m, str(dst))


# ------------------------------------------------------------------ checks

def session(path_or_bytes):
    import onnxruntime as ort
    return ort.InferenceSession(path_or_bytes, providers=["CPUExecutionProvider"])


def resized_original(src, w, h):
    m = onnx.load(str(src))
    d = m.graph.input[0].type.tensor_type.shape.dim
    d[2].dim_value, d[3].dim_value = h, w
    del m.graph.value_info[:]
    for o in m.graph.output:
        o.type.tensor_type.shape.dim[1].dim_param = "n"
    return m.SerializeToString()


def check_yunet(src, dst, w, h, frames):
    import cv2
    new, orig = session(str(dst)), session(resized_original(src, w, h))
    names = [o.name for o in orig.get_outputs()]
    worst = 0.0
    cv_ious, cv_dscore, cv_missing, cv_count = [], [], 0, 0
    det = cv2.FaceDetectorYN.create(str(src), "", (w, h), 0.6, 0.3, 5000)
    rng = np.random.default_rng(0)
    inputs = [rng.uniform(0, 255, (1, 3, h, w)).astype(np.float32)]
    for f in frames:  # real frames, resized to fill the input (top-left), like FaceDetectorYN with this size
        img = cv2.resize(f, (w, h))
        inputs.append(img.transpose(2, 0, 1)[None].astype(np.float32))
        faces = det.detect(img)[1]
        faces = np.zeros((0, 15)) if faces is None else faces
        cv_count += len(faces)
        out = new.run(None, {"input": inputs[-1]})[0][0]
        keep = nms(out, 0.6, 0.3)
        for fc in faces:
            box = np.array([fc[0], fc[1], fc[0] + fc[2], fc[1] + fc[3]])
            ious = [iou(box, out[i, :4]) for i in keep]
            j = int(np.argmax(ious)) if ious else -1
            if j < 0 or ious[j] < 0.5:
                cv_missing += 1
                continue
            cv_ious.append(ious[j])
            cv_dscore.append(abs(fc[14] - out[keep[j], 4]))
    for x in inputs:
        ref = yunet_reference_decode(orig.run(None, {"input": x}), names, w, h)
        got = new.run(None, {"input": x})[0][0]
        order = np.argsort(-ref[:, 4], kind="stable")[:YUNET_TOPK]
        worst = max(worst, float(np.abs(ref[order] - got).max()))
    return {"max_abs_diff_vs_original_decoded": worst,
            "opencv_faces": cv_count, "opencv_faces_not_found": cv_missing,
            "opencv_box_iou_min": float(min(cv_ious)) if cv_ious else None,
            "opencv_score_diff_max": float(max(cv_dscore)) if cv_dscore else None}


def check_palm(src, dst):
    new, orig = session(str(dst)), session(str(src))
    rng = np.random.default_rng(1)
    worst = 0.0
    for _ in range(3):
        x = rng.uniform(0, 1, (1, PALM_INPUT, PALM_INPUT, 3)).astype(np.float32)
        reg, sc = orig.run(None, {"input_1": x})
        full = np.concatenate([sc[0], reg[0], palm_anchors()], 1)
        order = np.argsort(-full[:, 0], kind="stable")[:PALM_TOPK]
        worst = max(worst, float(np.abs(full[order] - new.run(None, {"input_1": x})[0][0]).max()))
    return {"max_abs_diff_vs_original": worst}


def check_handpose(src, dst):
    new, orig = session(str(dst)), session(str(src))
    x = np.random.default_rng(2).uniform(0, 1, (1, 224, 224, 3)).astype(np.float32)
    ref = np.concatenate([o.reshape(1, -1) for o in orig.run(None, {"input_1": x})], 1)
    return {"max_abs_diff_vs_original": float(np.abs(ref - new.run(None, {"input_1": x})[0]).max())}


def check_anchors_against_zoo(zoo_file):
    """Our anchors vs the table in OpenCV Zoo's mp_palmdet.py (parsed as text, not executed)."""
    import re
    txt = Path(zoo_file).read_text()
    nums = re.findall(r"\[([0-9.]+),\s*([0-9.]+)\]", txt[txt.index("def _load_anchors"):])
    zoo = np.array(nums, np.float32)
    return {"zoo_anchors": len(zoo), "max_abs_diff": float(np.abs(zoo - palm_anchors()).max())}


def iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def nms(rows, conf, thr):
    idx = [i for i in np.argsort(-rows[:, 4], kind="stable") if rows[i, 4] > conf]
    keep = []
    for i in idx:
        if all(iou(rows[i, :4], rows[j, :4]) <= thr for j in keep):
            keep.append(i)
    return keep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=str(ROOT / "models"))
    ap.add_argument("--clips", nargs="*", default=[str(Path.home() / "Downloads/benchmark_face/bench_results/webcam_clip.mp4"),
                                                   str(Path.home() / "Downloads/benchmark_face/tilt_results/tilt.mp4")])
    ap.add_argument("--zoo-palm-py", help="OpenCV Zoo mp_palmdet.py, to compare the anchor table")
    a = ap.parse_args()
    models = Path(a.models)
    models.mkdir(parents=True, exist_ok=True)
    fetch(models)

    frames = []
    try:
        import cv2
        for c in a.clips:
            cap = cv2.VideoCapture(c)
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in np.linspace(0, max(n - 1, 0), 8).astype(int):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, f = cap.read()
                if ok:
                    frames.append(f)
    except ImportError:
        pass

    report = {"files": [], "checks": {}}
    yunet_src = models / "face_detection_yunet_2023mar.onnx"
    for w, h in YUNET_SIZES:
        dst = models / ("yunet_%dx%d.onnx" % (w, h))
        make_yunet(yunet_src, dst, w, h)
        report["checks"][dst.name] = check_yunet(yunet_src, dst, w, h, frames)
    palm = models / "palm_192.onnx"
    make_palm(models / "palm_detection_mediapipe_2023feb.onnx", palm)
    report["checks"][palm.name] = check_palm(models / "palm_detection_mediapipe_2023feb.onnx", palm)
    if a.zoo_palm_py:
        report["checks"]["palm_anchors_vs_zoo"] = check_anchors_against_zoo(a.zoo_palm_py)
    hand = models / "handpose_224.onnx"
    make_handpose(models / "handpose_estimation_mediapipe_2023feb.onnx", hand)
    report["checks"][hand.name] = check_handpose(models / "handpose_estimation_mediapipe_2023feb.onnx", hand)

    ok = True
    for p in [models / ("yunet_%dx%d.onnx" % s) for s in YUNET_SIZES] + [palm, hand]:
        r = trt85_check.inspect(p)
        report["files"].append(r)
        ok &= not r["problems"]
        print("%-20s in %s out %s opset %d TRT 8.5: %s" % (p.name, r["inputs"][0][1], r["outputs"][0][1], r["opset"],
                                                          "OK" if not r["problems"] else r["problems"]))
    for k, v in report["checks"].items():
        print("  check %-22s %s" % (k, v))
    (models / "face_hand_report.json").write_text(json.dumps(report, indent=1))
    print("wrote", models / "face_hand_report.json")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
