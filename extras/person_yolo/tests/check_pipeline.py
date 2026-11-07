"""Desktop check: the robot's letterbox (BGR and NV12) and output decoding, run on the CPU, against
Ultralytics/OpenCV (yolo_ref) and ONNX Runtime.

    ~/Downloads/benchmark_face/.venv/bin/python tests/check_pipeline.py \
        --clips ~/Downloads/benchmark_face/tilt_results/tilt.mp4 ~/Downloads/benchmark_face/bench_results/webcam_clip.mp4

  1. input tensor: C++ letterbox_pixel() vs cv2.resize INTER_LINEAR_EXACT (platform-independent) + pad 114
     (BGR frames), and vs
     cv2.cvtColor(NV12 -> BGR) + the same (NV12 frames); differences in 8-bit grey levels
  2. decoding: ONNX Runtime output -> C++ Decoder vs yolo_ref decode + un-letterbox (both heads)
  3. end to end: detections from the C++ tensor vs from the Ultralytics-style tensor
  4. --hw-resize: the NV12 frame scaled down first (as the VIC does; simulated with a box filter and with
     plain bilinear), letterboxed at scale 1, boxes mapped back to camera pixels (for_camera), vs the
     full-size NV12 path; and for_camera() with an unscaled frame must change nothing
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # extras/person_yolo
sys.path.insert(0, str(ROOT / "scripts"))
import yolo_ref as R  # noqa: E402


def bgr_to_nv12(f):
    h, w = f.shape[:2]
    i420 = cv2.cvtColor(f, cv2.COLOR_BGR2YUV_I420).reshape(-1)
    y, u, v = i420[: w * h], i420[w * h: w * h * 5 // 4], i420[w * h * 5 // 4:]
    uv = np.empty(w * h // 2, np.uint8)
    uv[0::2], uv[1::2] = u, v
    return np.concatenate([y, uv]).reshape(h * 3 // 2, w)


def scale_nv12(nv12, w, h, fw, fh, interp):
    """NV12 w x h -> fw x fh, each plane on its own (what a hardware scaler does with NV12)."""
    y = cv2.resize(nv12[:h], (fw, fh), interpolation=interp)
    uv = nv12[h:].reshape(h // 2, w // 2, 2)
    uv = cv2.resize(uv, (fw // 2, fh // 2), interpolation=interp)
    return np.concatenate([y, uv.reshape(fh // 2, fw)])


def cpp_pre(binary, work, img, fmt, dw, dh):
    raw, out = work / "frame.raw", work / "tensor.f32"
    img.tofile(raw)
    h = img.shape[0] if fmt == "bgr" else img.shape[0] * 2 // 3
    w = img.shape[1]
    subprocess.run([str(binary), "pre", str(raw), str(w), str(h), fmt, str(dw), str(dh), str(out)],
                   check=True, capture_output=True)
    return np.fromfile(out, np.float32).reshape(1, 3, dh, dw)


def cpp_post(binary, work, out, w, h, dw, dh, conf, frame=None):
    o, d = work / "out.f32", work / "dets.txt"
    out.astype(np.float32).tofile(o)
    subprocess.run([str(binary), "post", str(o), *map(str, out.shape), str(w), str(h), str(dw), str(dh), str(conf),
                    str(d), *([str(frame[0]), str(frame[1])] if frame else [])], check=True)
    txt = d.read_text().strip()
    return np.array([list(map(float, ln.split())) for ln in txt.split("\n")]) if txt else np.zeros((0, 5))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--frames", type=int, default=10, help="frames per clip")
    ap.add_argument("--bin", default=str(ROOT / "build/desktop/test_pipeline"))
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    work = ROOT / "build/desktop"
    work.mkdir(parents=True, exist_ok=True)
    frames = []
    for c in args.clips:
        frames += R.read_frames(c, n=args.frames)
    ok = True

    print("1. input tensor (C++ vs OpenCV INTER_LINEAR_EXACT), in 8-bit levels")
    sizes = [(640, 384), (640, 640), (416, 416), (416, 256), (320, 320), (320, 192)]
    for dw, dh in sizes:
        for fmt in ("bgr", "nv12"):
            maxd, frac, ul_frac = 0.0, [], []
            for f in frames:
                if fmt == "bgr":
                    img, ref_bgr = f, f
                else:
                    img = bgr_to_nv12(f)
                    ref_bgr = cv2.cvtColor(img, cv2.COLOR_YUV2BGR_NV12)
                ref = R.to_tensor(R.letterbox(ref_bgr, dw, dh, exact=True))  # OpenCV bit-exact bilinear
                got = cpp_pre(args.bin, work, img, fmt, dw, dh)
                ul = R.to_tensor(R.letterbox(ref_bgr, dw, dh))  # what Ultralytics gets with this cv2 build
                ul_frac.append(float((np.abs(got - ul) * 255 > 0.5).mean()))
                d = np.abs(got - ref) * 255
                maxd = max(maxd, float(d.max()))
                frac.append(float((d > 0.5).mean()))
            exact = (dw, dh) in [(640, 384), (640, 640), (320, 320), (320, 192)]  # scale 1/2 or 1/4
            good = maxd <= (0.01 if exact else 1.01)
            ok &= good
            print(f"   {dw}x{dh} {fmt:4s}: max diff {maxd:.3f} levels, values differing {100 * np.mean(frac):.3f}%"
                  f"  {'OK' if good else 'FAIL'}{' (exact expected)' if exact else ' (<=1 level expected)'}"
                  f"   [vs this Mac's 3-channel cv2.resize: {100 * np.mean(ul_frac):.1f}% differ by 1]")

    print("2/3. decoding and end-to-end detections")
    for model in ["yolo26n_640x384", "yolo26n_640x640", "yolo26n-raw_640x384", "yolo11n_640x384"]:
        sess = ort.InferenceSession(str(ROOT / "models" / f"{model}.onnx"), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dh, dw = inp.shape[2], inp.shape[3]
        dec_iou, dec_n, e2e_iou, e2e_n, dec_bad, e2e_bad, dconf = [], 0, [], 0, 0, 0, 0.0
        for f in frames:
            h, w = f.shape[:2]
            ref_t = R.to_tensor(R.letterbox(f, dw, dh))
            out = sess.run(None, {inp.name: ref_t})[0]
            ref = R.decode(out, args.conf)
            if len(ref):
                ref[:, :4] = R.unletterbox(ref[:, :4], w, h, dw, dh)
            got = cpp_post(args.bin, work, out, w, h, dw, dh, args.conf)
            m = R.match(ref, got, 0.99)
            dec_bad += len(m) != len(ref) or len(got) != len(ref)
            dec_n += len(ref)
            dec_iou += [x[2] for x in m]
            dconf = max([dconf] + [abs(ref[i, 4] - got[j, 4]) for i, j, _ in m])
            # end to end: C++ tensor -> ORT -> C++ decode, vs reference
            got_t = cpp_pre(args.bin, work, f, "bgr", dw, dh)
            got2 = cpp_post(args.bin, work, sess.run(None, {inp.name: got_t})[0], w, h, dw, dh, args.conf)
            m2 = R.match(ref, got2, 0.5)
            e2e_bad += len(m2) != len(ref) or len(got2) != len(ref)
            e2e_n += len(ref)
            e2e_iou += [x[2] for x in m2]
        good = dec_bad == 0 and min(dec_iou, default=1) > 0.9999
        ok &= good
        print(f"   {model:22s} decode: {dec_n} boxes, frames differing {dec_bad}, min IoU {min(dec_iou, default=1):.6f}, "
              f"max dconf {dconf:.1e} {'OK' if good else 'FAIL'} | end-to-end: frames differing {e2e_bad}, "
              f"mean IoU {np.mean(e2e_iou) if e2e_iou else 1:.5f}")
    print("4. --hw-resize (VIC scaling simulated) vs the full-size NV12 path, NMS-free head")
    for model in ["yolo26n_640x384", "yolo26n_416x256", "yolo26n_320x192"]:
        sess = ort.InferenceSession(str(ROOT / "models" / f"{model}.onnx"), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0]
        dh, dw = inp.shape[2], inp.shape[3]
        for name, interp in (("box filter", cv2.INTER_AREA), ("bilinear", cv2.INTER_LINEAR)):
            ref_all, got_all, same_map = [], [], True
            for f in frames:
                h, w = f.shape[:2]
                nv = bgr_to_nv12(f)
                out = sess.run(None, {inp.name: cpp_pre(args.bin, work, nv, "nv12", dw, dh)})[0]
                ref = cpp_post(args.bin, work, out, w, h, dw, dh, args.conf)
                same = cpp_post(args.bin, work, out, w, h, dw, dh, args.conf, frame=(w, h))
                same_map &= ref.shape == same.shape and bool(np.array_equal(ref, same))
                r = min(dw / w, dh / h)
                fw, fh = int(round(w * r)) & ~1, int(round(h * r)) & ~1
                small = scale_nv12(nv, w, h, fw, fh, interp)
                out2 = sess.run(None, {inp.name: cpp_pre(args.bin, work, small, "nv12", dw, dh)})[0]
                got_all.append(cpp_post(args.bin, work, out2, w, h, dw, dh, args.conf, frame=(fw, fh)))
                ref_all.append(ref)
            n_ref = sum(len(x) for x in ref_all)
            n_got = sum(len(x) for x in got_all)
            ms = [R.match(a, b, 0.5) for a, b in zip(ref_all, got_all)]
            ious = [m[2] for mm in ms for m in mm]
            top_iou = [R.match(a[:1], b[:1], 0.0)[0][2] if len(a) and len(b) else 0.0
                       for a, b in zip(ref_all, got_all) if len(a)]
            good = same_map and (not ref_all or np.mean(top_iou) > 0.9)
            ok &= good
            print(f"   {model:16s} {fw}x{fh} {name:10s}: boxes {n_ref} -> {n_got}, matched {len(ious)}, "
                  f"mean IoU {np.mean(ious) if ious else 0:.4f}, top-box IoU mean {np.mean(top_iou):.4f} "
                  f"min {np.min(top_iou):.4f}; unscaled mapping unchanged: {same_map}  {'OK' if good else 'FAIL'}")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
