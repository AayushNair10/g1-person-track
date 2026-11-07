"""Desktop check: the C++ OC-SORT gives the same tracks as Ultralytics' OC-SORT on the same detections.

    ~/Downloads/benchmark_face/.venv/bin/python tests/check_tracker.py \
        --clips ~/Downloads/benchmark_face/tilt_results/tilt.mp4 ~/Downloads/benchmark_face/bench_results/webcam_clip.mp4 \
        --models models/yolo26n_640x384.onnx models/yolo26n-raw_640x384.onnx

For each clip and model: frames resized to 1280x720, detections from ONNX Runtime through yolo_ref (the C++
detection path), at 30 FPS (every frame) and 15 FPS (every 2nd frame, like the head camera). The same
detections go into Ultralytics OCSORT (ocsort.yaml defaults) and into build/test_tracker (C++). Reports per
run: frames where the track sets differ, max box difference, and target ID changes / frames without target.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("YOLO_OFFLINE", "1")

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]  # extras/person_yolo
sys.path.insert(0, str(ROOT / "scripts"))
import yolo_ref as R  # noqa: E402


def detections(clip, model, cache_dir):
    """Per-frame (N, 6) [x1 y1 x2 y2 score cls] in 1280x720 pixels for every frame of the clip (cached)."""
    cache = cache_dir / f"dets_{Path(clip).stem}_{Path(model).stem}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["dets"])
    sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    in_h, in_w = inp.shape[2], inp.shape[3]
    out = []
    for f in R.read_frames(clip):
        h, w = f.shape[:2]
        d = R.decode(sess.run(None, {inp.name: R.to_tensor(R.letterbox(f, in_w, in_h))})[0])
        if len(d):
            d[:, :4] = R.unletterbox(d[:, :4], w, h, in_w, in_h)
        rows = np.zeros((len(d), 6), np.float32)
        rows[:, :5] = d
        out.append(rows)
    np.savez(cache, dets=np.array(out, dtype=object))
    return out


def run_python(dets):
    from ultralytics.engine.results import Boxes
    from ultralytics.trackers.oc_sort import OCSORT
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**YAML.load(check_yaml("ocsort.yaml")))
    tr = OCSORT(args=cfg)
    res = []
    for d in dets:
        t = tr.update(Boxes(d.reshape(-1, 6), orig_shape=(720, 1280)), None)
        res.append(np.asarray(t, np.float64).reshape(-1, 8))  # x1 y1 x2 y2 id score cls idx
    return res


def run_cpp(dets, binary, work):
    dfile, tfile = work / "dets.txt", work / "tracks.txt"
    with open(dfile, "w") as f:
        for i, d in enumerate(dets):
            f.write(f"F {i} {len(d)}\n")
            for r in d:
                f.write(f"{r[0]:.9g} {r[1]:.9g} {r[2]:.9g} {r[3]:.9g} {r[4]:.9g} {int(r[5])}\n")
    summary = subprocess.run([str(binary), str(dfile), str(tfile)], check=True, capture_output=True, text=True).stdout
    res, lines = [], open(tfile).read().split("\n")
    k = 0
    while k < len(lines) and lines[k].startswith("F "):
        _, _, n, _tid = lines[k].split()
        rows = [list(map(float, lines[k + 1 + j].split())) for j in range(int(n))]
        res.append(np.array(rows, np.float64).reshape(-1, 7))  # id x1 y1 x2 y2 score idx
        k += 1 + int(n)
    return res, summary.strip()


def compare(py, cpp):
    bad, max_box = 0, 0.0
    first_bad = None
    for i, (p, c) in enumerate(zip(py, cpp)):
        pid = sorted(int(x) for x in p[:, 4])
        cid = sorted(int(x) for x in c[:, 0])
        if pid != cid:
            bad += 1
            first_bad = first_bad if first_bad is not None else (i, pid, cid)
            continue
        for row in p:
            q = c[c[:, 0] == row[4]][0]
            max_box = max(max_box, float(np.abs(row[:4] - q[1:5]).max()))
    return bad, max_box, first_bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clips", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=[str(ROOT / "models/yolo26n_640x384.onnx")])
    ap.add_argument("--bin", default=str(ROOT / "build/desktop/test_tracker"))
    ap.add_argument("--work", default=str(ROOT / "build/desktop"))
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for model in args.models:
        for clip in args.clips:
            dets_all = detections(clip, model, work)
            for stride in (1, 2):
                dets = dets_all[::stride]
                py = run_python(dets)
                cpp, summary = run_cpp(dets, args.bin, work)
                bad, max_box, first_bad = compare(py, cpp)
                all_ok &= bad == 0
                n_ids_py = len({int(x) for p in py for x in p[:, 4]})
                print(f"{Path(model).stem:22s} {Path(clip).stem:12s} {30 // stride:2d} FPS: {len(dets)} frames, "
                      f"{sum(len(d) for d in dets)} dets, {n_ids_py} IDs | C++ vs Python: {bad} frames differ, "
                      f"max box diff {max_box:.2e} px | C++ target: {summary}", flush=True)
                if first_bad:
                    print(f"    first difference at frame {first_bad[0]}: python ids {first_bad[1]} c++ ids {first_bad[2]}")
    print("ALL MATCH" if all_ok else "MISMATCH")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
