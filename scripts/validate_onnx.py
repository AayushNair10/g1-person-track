"""Phase 2 (laptop): check every ONNX file with ONNX Runtime against the PyTorch .pt on real frames.

    ~/Downloads/benchmark_face/.venv/bin/python scripts/validate_onnx.py \
        --clips ~/Downloads/benchmark_face/tilt_results/tilt.mp4 ~/Downloads/benchmark_face/bench_results/webcam_clip.mp4

Frames are resized to 1280x720 (the robot's stream size). Two checks per model:
  A. export numerics: the same letterboxed uint8 image goes through the .pt (Ultralytics, PyTorch CPU) and
     through the ONNX (ONNX Runtime CPU, decoded by yolo_ref = what the C++ does); person boxes are matched.
     The .pt runs the same head as the ONNX: nms=False (NMS-free head) for yolo26n_*, the default raw head +
     NMS for yolo26n-raw_* and yolo11n_*. (Ultralytics 8.4 .pt predict uses the raw head unless nms=False.)
  B. full pipeline: frame -> yolo_ref letterbox -> ONNX -> decode -> un-letterbox, against Ultralytics
     predicting on the same ONNX file from the 1280x720 frame (checks letterbox + box mapping + NMS).
Writes models/onnx_validation.md.
"""
import argparse
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import yolo_ref as R  # noqa: E402

PT_FOR = {"yolo26n": "yolo26n.pt", "yolo26n-raw": "yolo26n.pt", "yolo11n": "yolo11n.pt"}


def compare(ref, test):
    """ref, test: lists (per frame) of (N,5) arrays. Returns agreement stats."""
    n_ref = n_test = n_match = same_count = 0
    ious, dconf = [], []
    for a, b in zip(ref, test):
        n_ref += len(a)
        n_test += len(b)
        same_count += len(a) == len(b)
        for i, j, iou in R.match(a, b):
            n_match += 1
            ious.append(iou)
            dconf.append(abs(a[i, 4] - b[j, 4]))
    return {
        "ref": n_ref, "test": n_test, "matched": n_match,
        "recall": n_match / max(n_ref, 1), "precision": n_match / max(n_test, 1),
        "iou_mean": float(np.mean(ious)) if ious else float("nan"),
        "iou_min": float(np.min(ious)) if ious else float("nan"),
        "dconf_max": float(np.max(dconf)) if dconf else float("nan"),
        "same_count": same_count / max(len(ref), 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--models", default="models")
    ap.add_argument("--frames", type=int, default=20, help="frames per clip, evenly spaced")
    ap.add_argument("--conf", type=float, default=0.25)
    args = ap.parse_args()
    from ultralytics import YOLO

    frames = []
    for c in args.clips:
        frames += R.read_frames(c, n=args.frames)
    print(f"{len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}")
    mdir = Path(args.models)
    pts = {}
    rows = []
    for onnx_path in sorted(mdir.glob("*.onnx")):
        name, size = onnx_path.stem.rsplit("_", 1)
        in_w, in_h = map(int, size.split("x"))
        pt_name = PT_FOR[name]
        if pt_name not in pts:
            pts[pt_name] = YOLO(str(mdir / pt_name))
        pt = pts[pt_name]
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp = sess.get_inputs()[0].name
        ul_onnx = YOLO(str(onnx_path), task="detect")
        nms_free = sess.get_outputs()[0].shape[-1] == 6

        ref_a, test_a, ref_b, test_b = [], [], [], []
        t_ort = []
        for f in frames:
            h, w = f.shape[:2]
            lb = R.letterbox(f, in_w, in_h)
            # A: PyTorch .pt on the letterboxed image (already model size, so Ultralytics adds no padding)
            r = pt.predict(lb, imgsz=(in_h, in_w), conf=args.conf, classes=[R.PERSON], device="cpu", verbose=False,
                           **({"nms": False} if nms_free else {}))[0]
            ref_a.append(np.concatenate([r.boxes.xyxy.numpy(), r.boxes.conf.numpy()[:, None]], 1).astype(np.float64))
            t0 = time.perf_counter()
            out = sess.run(None, {inp: R.to_tensor(lb)})[0]
            t_ort.append(time.perf_counter() - t0)
            d = R.decode(out, args.conf)
            test_a.append(d)
            # B: full pipeline in frame coordinates vs Ultralytics on the same ONNX
            dd = d.copy()
            if len(dd):
                dd[:, :4] = R.unletterbox(dd[:, :4], w, h, in_w, in_h)
            test_b.append(dd)
            rb = ul_onnx.predict(f, imgsz=(in_h, in_w), conf=args.conf, classes=[R.PERSON], device="cpu", verbose=False)[0]
            ref_b.append(np.concatenate([rb.boxes.xyxy.numpy(), rb.boxes.conf.numpy()[:, None]], 1).astype(np.float64))
        a, b = compare(ref_a, test_a), compare(ref_b, test_b)
        rows.append((onnx_path.name, a, b, 1000 * float(np.median(t_ort))))
        print(f"{onnx_path.name:26s} A(.pt vs onnx): recall {a['recall']:.3f} prec {a['precision']:.3f} "
              f"IoU mean {a['iou_mean']:.4f} min {a['iou_min']:.4f} |dconf| max {a['dconf_max']:.4f} | "
              f"B(pipeline vs Ultralytics-onnx): IoU mean {b['iou_mean']:.5f} min {b['iou_min']:.5f} "
              f"recall {b['recall']:.3f} prec {b['precision']:.3f} | ORT CPU {rows[-1][3]:.1f} ms", flush=True)

    md = ["# ONNX validation (laptop, ONNX Runtime CPU vs PyTorch)", "",
          f"{len(frames)} frames from {', '.join(Path(c).name for c in args.clips)} resized to 1280x720; "
          f"person class, conf > {args.conf}. Boxes matched one-to-one at IoU >= 0.5.", "",
          "A = same letterboxed image through the .pt and the ONNX, same head (export numerics): `yolo26n_*` = "
          "NMS-free head (.pt with nms=False), `yolo26n-raw_*` / `yolo11n_*` = raw head + NMS (.pt default).  ",
          "B = our letterbox + decode + un-letterbox (= the C++ path) vs Ultralytics predicting with the same ONNX.", "",
          "| file | A recall | A precision | A IoU mean | A IoU min | A max abs(dconf) | B IoU mean | B recall | B precision | ORT CPU ms |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for n, a, b, t in rows:
        md.append(f"| {n} | {a['recall']:.3f} | {a['precision']:.3f} | {a['iou_mean']:.4f} | {a['iou_min']:.4f} | "
                  f"{a['dconf_max']:.4f} | {b['iou_mean']:.5f} | {b['recall']:.3f} | {b['precision']:.3f} | {t:.1f} |")
    (mdir / "onnx_validation.md").write_text("\n".join(md) + "\n")
    print(f"\nwrote {mdir / 'onnx_validation.md'}")


if __name__ == "__main__":
    main()
