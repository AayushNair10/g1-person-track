"""Desktop check: the robot's face + hand C++ code (CPU build of the same functions the GPU kernels and the app
use) against the Python reference scripts/facehand_ref.py, on frames of your clips with real model outputs
(ONNX Runtime), plus synthetic 3-D hands for all 15 gestures.

    clang++ -std=c++17 -O2 tests/desktop/test_facehand.cpp src/gestures.cpp -o build/desktop/test_facehand
    ~/Downloads/benchmark_face/.venv/bin/python tests/desktop/check_facehand.py --clips CLIP [CLIP ...]

  1. input tensors: face letterbox (whole frame) and palm letterbox (region), BGR and NV12; rotated hand crops
  2. face decoding (YuNet output) and palm decoding + hand crop (palm output), on real model outputs
  3. hand landmarks -> frame, next crop, finger states, gesture, on real model outputs
  4. gesture rules on synthetic hands: each of the 15 gestures, turned and scaled
"""
import argparse
import math
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import facehand_ref as R  # noqa: E402

BIN = ROOT / "build/desktop/test_facehand"
WORK = ROOT / "build/desktop"


def run(*args):
    subprocess.run([str(BIN), *map(str, args)], check=True, capture_output=True)


def bgr_to_nv12(f):
    h, w = f.shape[:2]
    i420 = cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y, u, v = i420[: w * h], i420[w * h: w * h * 5 // 4], i420[w * h * 5 // 4:]
    uv = np.empty(w * h // 2, np.uint8)
    uv[0::2], uv[1::2] = u, v
    return np.concatenate([y, uv]).reshape(h * 3 // 2, w)


def cpp_letterbox(img, fmt, w, h, lb, nhwc, rgb, scale):
    img.tofile(WORK / "frame.raw")
    run("letterbox", WORK / "frame.raw", w, h, fmt, lb["x0"], lb["y0"], lb["w"], lb["h"], lb["dw"], lb["dh"],
        int(nhwc), int(rgb), scale, WORK / "t.f32")
    t = np.fromfile(WORK / "t.f32", np.float32)
    return t.reshape((1, lb["dh"], lb["dw"], 3) if nhwc else (1, 3, lb["dh"], lb["dw"]))


def cpp_rotcrop(img, fmt, w, h, r, out):
    img.tofile(WORK / "frame.raw")
    run("rotcrop", WORK / "frame.raw", w, h, fmt, *("%.9g" % v for v in r), out, WORK / "t.f32")
    return np.fromfile(WORK / "t.f32", np.float32).reshape(1, out, out, 3)


def load_rows(path):
    txt = Path(path).read_text().strip()
    return np.array([list(map(float, ln.split())) for ln in txt.split("\n")]) if txt else np.zeros((0, 0))


# --------------------------------------------------------------------------- synthetic hands

BASES = {1: (-0.020, -0.025), 2: (-0.022, -0.085), 3: (-0.002, -0.090), 4: (0.018, -0.085), 5: (0.035, -0.075)}
LENGTHS = {1: (0.035, 0.030, 0.025), 2: (0.040, 0.025, 0.020), 3: (0.045, 0.028, 0.022), 4: (0.042, 0.026, 0.021),
           5: (0.032, 0.020, 0.018)}
STRAIGHT, CURLED = (0, 5, 5), (70, 100, 60)


def chain(base, direction, lengths, bends):
    """Joints along a finger bending towards the palm (+z) by `bends` degrees at each joint."""
    pts, p = [np.array(base, float)], np.array(base, float)
    d = np.array(direction, float) / np.linalg.norm(direction)
    total = 0.0
    for L, b in zip(lengths, bends):
        total += math.radians(b)
        # rotate d's in-plane direction towards +z by the accumulated bend
        dd = np.array([d[0] * math.cos(total), d[1] * math.cos(total), math.sin(total) * np.linalg.norm(d[:2])])
        p = p + dd / np.linalg.norm(dd) * L
        pts.append(p)
    return pts


def synth_hand(fingers, thumb="tucked", pinch=False):
    """fingers: 4 states for index..little ('s' straight, 'c' curled). thumb: 'out' | 'tucked'."""
    W = np.zeros((21, 3))
    for k, st in zip((2, 3, 4, 5), fingers):
        b = BASES[k]
        spread = (b[0] + 0.002) * 1.5  # fingers fan out a little
        pts = chain((b[0], b[1], 0.0), (spread, -1.0, 0.0), LENGTHS[k], STRAIGHT if st == "s" else CURLED)
        W[1 + 4 * (k - 1): 5 + 4 * (k - 1)] = pts
    cmc = np.array([*BASES[1], 0.0])
    if pinch:  # index half bent; thumb tip meets the index tip
        W[5:9] = chain((BASES[2][0], BASES[2][1], 0.0), (-0.03, -1.0, 0.0), LENGTHS[2], (35, 45, 30))
        P = W[8] + np.array([0.004, 0.0, 0.0])
        W[1], W[2], W[3], W[4] = cmc, cmc + (P - cmc) * 0.4 + [-0.01, 0, 0], cmc + (P - cmc) * 0.75 + [-0.005, 0, 0], P
    elif thumb == "out":
        W[1:5] = chain(cmc, (-0.9, -0.44, 0.0), LENGTHS[1], STRAIGHT)
    else:
        W[1:5] = chain(cmc, (0.55, -0.55, 0.0), LENGTHS[1], (20, 40, 30))
    return W


GESTURES = {  # label: (fingers, thumb, pinch, image direction of the vector that must point somewhere)
    "Open palm": ("ssss", "out", False, None),
    "Fist": ("cccc", "tucked", False, None),
    "Thumbs up": ("cccc", "out", False, ("thumb", "up")),
    "Thumbs down": ("cccc", "out", False, ("thumb", "down")),
    "Pointing up": ("sccc", "tucked", False, ("index", "up")),
    "Pointing down": ("sccc", "tucked", False, ("index", "down")),
    "Pointing left": ("sccc", "tucked", False, ("index", "left")),
    "Pointing right": ("sccc", "tucked", False, ("index", "right")),
    "Victory": ("sscc", "tucked", False, None),
    "Three": ("sssc", "tucked", False, None),
    "OK": ("ssss", "tucked", True, None),
    "Pinch": ("sccc", "tucked", True, None),
    "Rock": ("sccs", "tucked", False, None),
    "I love you": ("sccs", "out", False, None),
    "Call me": ("cccs", "out", False, None),
}
DIRS = {"up": -90.0, "down": 90.0, "left": 180.0, "right": 0.0}


def image_points(W, turn_deg, scale=2500.0, at=(640.0, 400.0)):
    c, s = math.cos(math.radians(turn_deg)), math.sin(math.radians(turn_deg))
    xy = W[:, :2] @ np.array([[c, s], [-s, c]])
    return xy * scale + at


def synth_cases():
    cases = []
    for label, (fingers, thumb, pinch, direction) in GESTURES.items():
        W = synth_hand(fingers, thumb, pinch)
        if direction:
            a, b = (2, 4) if direction[0] == "thumb" else (5, 8)
            v = W[b, :2] - W[a, :2]
            base = DIRS[direction[1]] - math.degrees(math.atan2(v[1], v[0]))
            turns = [base - 12, base, base + 12]
        else:
            turns = [-30, 0, 25]
        for t in turns:
            for sc in (1500.0, 4000.0):
                cases.append((label, W, image_points(W, t, sc)))
    return cases


# --------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=6, help="frames per clip")
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    models = ROOT / "models"
    face = ort.InferenceSession(str(models / "yunet_640x384.onnx"), providers=["CPUExecutionProvider"])
    face320 = ort.InferenceSession(str(models / "yunet_320x192.onnx"), providers=["CPUExecutionProvider"])
    palm = ort.InferenceSession(str(models / "palm_192.onnx"), providers=["CPUExecutionProvider"])
    hand = ort.InferenceSession(str(models / "handpose_224.onnx"), providers=["CPUExecutionProvider"])
    frames = []
    for c in a.clips:
        cap = cv2.VideoCapture(c)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for i in np.linspace(0, n - 1, a.frames).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, f = cap.read()
            if ok:
                frames.append(cv2.resize(f, (1280, 720), interpolation=cv2.INTER_AREA))
    W, H = 1280, 720
    ok_all = True

    def report(name, good, detail):
        nonlocal ok_all
        ok_all &= good
        print("  %-58s %s  %s" % (name, "OK  " if good else "FAIL", detail))

    print("1. input tensors (C++ vs reference)")
    rng = np.random.default_rng(0)
    worst = {}
    for f in frames:
        nv = bgr_to_nv12(f)
        f_nv = cv2.cvtColor(nv, cv2.COLOR_YUV2BGR_NV12)
        for (dw, dh) in ((640, 384), (320, 192), (416, 256)):
            lb = R.make_letterbox(0, 0, W, H, dw, dh)
            for fmt, img, ref_img in (("bgr", f, f), ("nv12", nv, f_nv)):
                d = np.abs(cpp_letterbox(img, fmt, W, H, lb, False, False, 1.0) - R.letterbox(ref_img, lb, False, False, 1.0))
                k = "face %dx%d %s" % (dw, dh, fmt)
                worst[k] = max(worst.get(k, 0), float(d.max()))
        for _ in range(2):  # palm search regions of random size and place
            rw, rh = int(rng.integers(120, 700)), int(rng.integers(120, 700))
            x0, y0 = int(rng.integers(0, W - rw)), int(rng.integers(0, H - rh))
            lb = R.make_letterbox(x0, y0, rw, rh, 192, 192)
            d = np.abs(cpp_letterbox(f, "bgr", W, H, lb, True, True, 1 / 255) - R.letterbox(f, lb, True, True, 1 / 255)) * 255
            worst["palm region bgr"] = max(worst.get("palm region bgr", 0), float(d.max()))
        for _ in range(3):
            r = (float(rng.uniform(0, W)), float(rng.uniform(0, H)), float(rng.uniform(40, 800)), float(rng.uniform(-math.pi, math.pi)))
            d = np.abs(cpp_rotcrop(f, "bgr", W, H, r, 224) - R.rotcrop(f, r, 224)) * 255
            worst["hand crop bgr"] = max(worst.get("hand crop bgr", 0), float(d.max()))
            d = np.abs(cpp_rotcrop(nv, "nv12", W, H, r, 224) - R.rotcrop(f_nv, r, 224)) * 255
            worst["hand crop nv12"] = max(worst.get("hand crop nv12", 0), float(d.max()))
    for k, v in worst.items():
        exact = k.startswith("face 640") or k.startswith("face 320")
        lim = 0.0 if exact else (1.0 if "palm" in k or "416" in k else 0.01)
        report(k, v <= lim + 1e-3, "max diff %.4f levels (%s)" % (v, "exact expected" if exact else "<= %.2f" % lim))

    print("2. face and palm decoding on real model outputs")
    nf = nd = 0
    dmax = 0.0
    palm_rows = 0
    pmax = rmax = 0.0
    hand_cases = []
    for f in frames:
        for sess, (dw, dh) in ((face, (640, 384)), (face320, (320, 192))):
            lb = R.make_letterbox(0, 0, W, H, dw, dh)
            out = sess.run(None, {"input": R.letterbox(f, lb, False, False, 1.0)})[0][0]
            ref = R.decode_faces(out, lb, W, H)
            out.astype(np.float32).tofile(WORK / "o.f32")
            run("faces", WORK / "o.f32", 0, 0, W, H, dw, dh, W, H, 0.6, WORK / "r.txt")
            got = load_rows(WORK / "r.txt")
            nf += len(ref)
            nd += len(ref) != len(got)
            if len(ref) and len(got) == len(ref):
                dmax = max(dmax, float(np.abs(np.array(ref) - got).max()))
            if sess is face and ref:
                tf = max(ref, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]))
                x0, y0, rw, rh = R.search_region(tf, W, H)
                lbp = R.make_letterbox(x0, y0, rw, rh, 192, 192)
                po = palm.run(None, {"input_1": R.letterbox(f, lbp, True, True, 1 / 255)})[0][0]
                ref_p = R.decode_palms(po, lbp, conf=0.02)  # low threshold: exercise many rows
                po.astype(np.float32).tofile(WORK / "p.f32")
                run("palms", WORK / "p.f32", x0, y0, rw, rh, 0.02, WORK / "rp.txt")
                got_p = load_rows(WORK / "rp.txt")
                palm_rows += len(ref_p)
                if len(ref_p) != len(got_p):
                    pmax = float("inf")
                for rp, gp in zip(ref_p, got_p):
                    pmax = max(pmax, float(np.abs(rp - gp[:19]).max()))
                    roi = R.roi_from_palm(rp)
                    rmax = max(rmax, float(np.abs(np.array(roi[:3]) - gp[19:22]).max()), abs(roi[3] - gp[22]) * 100)
                    hand_cases.append((f, roi))
    report("faces: %d detections over %d frames x 2 sizes" % (nf, len(frames)), nd == 0 and dmax < 1e-3,
           "count mismatches %d, max diff %.2e px" % (nd, dmax))
    report("palms (threshold 0.02): %d candidates" % palm_rows, pmax < 1e-3, "max diff %.2e px" % pmax)
    report("hand crop from palm", rmax < 1e-3, "max diff %.2e (px, angle x100)" % rmax)

    print("3. hand landmarks -> frame, next crop, finger states, gesture (real model outputs)")
    lmax = nmax = 0.0
    shape_bad = gest_bad = 0
    for f, roi in hand_cases[:40]:
        o = hand.run(None, {"input_1": R.rotcrop(f, roi, 224)})[0][0]
        ref = R.decode_hand(o, roi)
        nxt = R.roi_from_landmarks(ref)
        shape, touch = R.hand_shape(ref)
        o.astype(np.float32).tofile(WORK / "h.f32")
        run("hand", WORK / "h.f32", *("%.9g" % v for v in roi), WORK / "rh.txt")
        lines = (WORK / "rh.txt").read_text().split("\n")
        got = np.array([list(map(float, ln.split())) for ln in lines[:21]])
        lmax = max(lmax, float(np.abs(got[:, :2] - ref["xy"]).max()))
        gn = list(map(float, lines[22].split()[1:]))
        nmax = max(nmax, float(np.abs(np.array(nxt[:3]) - gn[:3]).max()), abs(nxt[3] - gn[3]) * 100)
        gs = lines[23].split()
        shape_bad += [int(v) for v in gs[1:6]] != shape or abs(float(gs[6]) - touch) > 1e-3
        gest_bad += lines[24][len("gesture "):].strip() != R.classify(ref)
    report("landmarks to frame (%d hands)" % min(40, len(hand_cases)), lmax < 1e-2, "max diff %.2e px" % lmax)
    report("next crop from landmarks", nmax < 1e-2, "max diff %.2e (px, angle x100)" % nmax)
    report("finger states and gesture: C++ == reference", shape_bad == 0 and gest_bad == 0,
           "shape mismatches %d, gesture mismatches %d" % (shape_bad, gest_bad))

    print("4. gesture rules on synthetic hands (each gesture turned 3 ways x 2 sizes)")
    per = {}
    for label, Wd, xy in synth_cases():
        pts = np.c_[xy, Wd]
        np.savetxt(WORK / "g.txt", pts)
        out = subprocess.run([str(BIN), "gesture", str(WORK / "g.txt")], check=True, capture_output=True, text=True).stdout
        cpp = out.split("\n")[1][len("gesture "):].strip()
        py = R.classify(dict(xy=xy, world=Wd))
        good = per.setdefault(label, [0, 0, set()])
        good[0] += cpp == label and py == label
        good[1] += 1
        if cpp != label:
            good[2].add(cpp or "(none)")
    for label, (g, n, wrong) in per.items():
        report(label, g == n, "%d/%d%s" % (g, n, ("  got " + ", ".join(sorted(wrong))) if wrong else ""))
    print("ALL OK" if ok_all else "SOME CHECKS FAILED")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
