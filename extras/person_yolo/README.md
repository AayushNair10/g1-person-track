# Person tracking with YOLO26n (frozen)

Real-time single-person tracking on the Unitree G1's onboard Jetson (Orin NX 16 GB, JetPack 5.1.1) using the
head camera: **YOLO26n** (TensorRT fp16) finds people, a C++ **OC-SORT** keeps stable IDs, and one target person
is followed. Its box and ID are printed every frame. Lightweight first: C++ only, nothing installed on the robot
(CUDA 11.4, TensorRT 8.5.2, OpenCV 4.2 and GStreamer are already there).

This was the repository's first app; it now lives here, self-contained, so the face + gesture app in
the repository root can change without affecting it. It is not built by default.

```
head camera, RealSense D435i colour sensor read directly (/dev/video4, YUYV 640x360 @ 30 FPS; "Option B":
  Unitree's videohub_pc4 is stopped by hand first)
  -> camera thread (newest frame wins), frames straight into pinned memory
  -> one CUDA kernel: letterbox 640x360 -> 640x384, BGR/NV12 -> RGB, /255, CHW, into the TensorRT input
  -> TensorRT YOLO26n fp16 (enqueueV2, replayed as a CUDA graph)
  -> decode person boxes (NMS-free head: no NMS) -> C++ OC-SORT -> target selector
  -> stdout / CSV: frame, timestamp, target_id, x1, y1, x2, y2, conf + per-stage timings (+ annotated MP4)
```


## Build and run (on the robot)

Everything here is relative to this folder (`extras/person_yolo/`): its own `src/`, `Makefile`, `scripts/` and
`models/`. Copy the folder to the robot, then:

```bash
cd extras/person_yolo
scripts/build_engines.sh models/yolo26n_640x384.onnx      # models/*.onnx: from scripts/export_models.py (laptop)
scripts/build.sh                                            # -> build/person_track
sudo pkill -f videohub_pc4; build/person_track --print-every 30
```

## What was checked on the laptop

**ONNX export (Phase 2).** `scripts/export_models.py` exports fixed-size opset-16 ONNX files and checks each
against the op list of the TensorRT 8.5 ONNX parser (taken from the onnx-tensorrt `release/8.5-GA` source, not
the outdated docs table), plus TopK/Mod rules. All 10 files pass. No LayerNormalization; the NMS-free head's
TopK has a constant K and its Mod has integer inputs, both of which the TRT 8.5 parser supports.
`scripts/validate_onnx.py` compares each ONNX (ONNX Runtime) with the `.pt` (PyTorch) running the same head on
40 frames from your two clips: recall/precision 1.000, box IoU >= 0.994, score difference 0.0000 for every
file. Our letterbox + decode + box mapping (the C++ path) equals Ultralytics' own ONNX predictions (IoU 1.00000).
Full table: [models/onnx_validation.md](models/onnx_validation.md).

| file | input | output | role |
|---|---|---|---|
| `yolo26n_640x384` | 1x3x384x640 | 1x300x6 | **default**: NMS-free head; 16:9 input, same picture as 640x640 with ~40% less compute |
| `yolo26n_640x640`, `_416x416`, `_320x320` | square | 1x300x6 | the sizes in the plan |
| `yolo26n_416x256`, `_320x192` | 16:9 | 1x300x6 | faster, smaller |
| `yolo26n-raw_640x384`, `_640x640` | | 1x84xN | fallback 1: same weights, raw head + C++ NMS |
| `yolo11n_640x384`, `_640x640` | | 1x84xN | fallback 2 |

**Finding about the benchmark head.** In Ultralytics 8.4.174 a `.pt` model predicts with YOLO26's *raw head +
NMS* unless `nms=False` is passed, and the CoreML runs that would have used the NMS-free head failed. So the
Mac benchmark's "77.5 FPS, 0 ID switches" was measured with the raw head. On the clips the two heads give the
same target (scores 0.82 vs 0.83); they differ only on one faint background object scored 0.25-0.29. Both
are built: the NMS-free head is the default as planned, and the raw head is the first fallback. The app
detects which one an engine has.

**Tracker.** `src/ocsort.cpp` is a line-by-line C++ port of the exact OC-SORT the benchmark ran
(Ultralytics `ocsort.yaml`: 8-state Kalman on x/y/aspect/height, IoU x score cost + direction-consistency
cost, optimal assignment with a cost limit, observation-centric recovery and re-update, 30-frame buffer).
`tests/check_tracker.py` feeds identical detections to both on your two clips (2 heads x 2 clips x 30/15 FPS:
8 runs, 7,350 frames): **0 frames differ, boxes agree to 1e-4 px**. Over the benchmark's window (frames 15-315 of
`webcam_clip.mp4`) the target keeps one ID with 93% coverage, as in the benchmark (0 switches, 92.7%).

**Preprocessing and decoding.** The CUDA kernel's per-pixel code (`src/preprocess_math.hpp`) is shared
with a CPU build. `tests/check_pipeline.py` shows it is **bit-exact** against OpenCV's
platform-independent bilinear (`INTER_LINEAR_EXACT`) at 640x384 / 640x640 / 320x320 / 320x192, for BGR and
for NV12 frames, and within 1 grey level at 416 sizes. Decoding (both heads, with NMS) is identical to the
Python reference. End-to-end detections match with IoU >= 0.999.

**Compilation.** `../../tests/desktop/syntax_check.sh` (run on this folder's sources at the time) compiled every source (and both sides of the CUDA file)
against the real TensorRT 8.5, CUDA 11.4 and OpenCV 4.2.0 headers with `-Wall -Wextra -Wshadow`: no warnings.

## Frame rate and compute

Compute of the detector per frame (multiply-adds x 2, from the ONNX graphs) and, on your two clips (300 frames),
how the largest person's box compares with the 640x384 default (the robot clip decides, because people are
smaller at robot distances):

| input | GFLOPs | vs 640x384 on the laptop clips |
|---|---|---|
| 640x640 (the plan) | 5.48 | same: 0 missed, box IoU 0.988 |
| **640x384 (default)** | **3.26** | reference |
| 416x416 | 2.29 | |
| 416x256 | 1.40 | 1 of 286 missed, IoU 0.972 |
| 320x320 | 1.35 | |
| 320x192 | 0.81 | 4 of 286 missed, IoU 0.962 (5th percentile 0.891) |

Switches for lower compute, each compared on the robot with `run_live.sh` runs (runbook step 8):

- the camera mode itself (Option B): the default 640x360 is exactly the picture inside the 640x384 model input,
  so nothing is resized at all; larger modes (848x480, 1280x720 @ 30) see more detail but need scaling.
- `--hw-resize`: the VIC (Jetson's video scaler, separate from GPU and CPU) shrinks the frame to the model's
  picture size inside GStreamer (e.g. 1280x720 -> 640x360 NV12, 0.35 MB instead of 2.8 MB BGR), so the GPU
  kernel only pads. Boxes still come out in camera pixels. Laptop simulation: same target, box IoU >= 0.99 vs
  the full-size path (`tests/check_pipeline.py`, part 4).
- a smaller engine: 416x256 is 2.3x and 320x192 4x less compute than 640x384.
- lean engines (`LEAN=1 scripts/build_engines.sh`): no cuDNN/cuBLAS, so less RAM and a faster start.
- DLA engines (`DLA=0 scripts/build_engines.sh`): the network runs on the Orin's deep-learning accelerator, which
  leaves the GPU free.
- `--every N`: process 1 of every N frames (skipped frames are never copied), with the tracker's memory scaled
  to the same number of seconds.

With Option A (Unitree's stream) the camera is 15 FPS, so live output cannot go above 15 FPS with Option A. What can be optimised is
keeping every frame and cutting the delay from frame arrival to result. Done:

- 16:9 model input (640x384) instead of 640x640: same picture, ~40% less compute
- one CUDA kernel for the whole preprocessing, writing straight into the TensorRT input
- frames decoded directly into pinned memory (no extra CPU copies), one GPU wait per frame
- TensorRT inference recorded once as a CUDA graph and replayed
- camera on its own thread, newest frame wins; video encoding on its own thread
- `videohub-nv12` source: NV12 straight from the hardware decoder, colour conversion done on the GPU, skipping
  the CPU `videoconvert` in the lab pipeline (bit-exact with OpenCV's NV12->BGR + resize)
- switches to measure on the robot: `--zero-copy`, `--no-graph`, `--dec-opts "enable-max-performance=1"`
  (the last one raises the decoder clock: lower latency, slightly more power)

`scripts/run_offline.sh` measures the pipeline's maximum rate without the camera limit. Going above 15 FPS live
needs Option B: the RealSense colour sensor read directly, at 30 FPS, after `videohub_pc4` has been stopped by
hand (the program never stops it). `--source /dev/video4 --width 640 --height 360 --fps 30` (OpenCV V4L2, as the
lab guide says) or add `--format nv12` (GStreamer `v4l2src`, the VIC converts YUYV -> NV12). Steps:
[docs/RUNBOOK.md](docs/RUNBOOK.md) section 7.

## Reproducing the laptop checks

```bash
PY=~/Downloads/benchmark_face/.venv/bin/python
CLIPS="$HOME/Downloads/benchmark_face/tilt_results/tilt.mp4 $HOME/Downloads/benchmark_face/bench_results/webcam_clip.mp4"
$PY scripts/export_models.py                      # models/*.onnx + onnx_report.json
$PY scripts/validate_onnx.py --clips $CLIPS       # models/onnx_validation.md
mkdir -p build/desktop
clang++ -std=c++17 -O2 tests/test_tracker.cpp src/ocsort.cpp -o build/desktop/test_tracker
clang++ -std=c++17 -O2 tests/test_pipeline.cpp -o build/desktop/test_pipeline
$PY tests/check_tracker.py --clips $CLIPS --models models/yolo26n_640x384.onnx models/yolo26n-raw_640x384.onnx
$PY tests/check_pipeline.py --clips $CLIPS
```

## Notes

- The OC-SORT port and the YOLO26/YOLO11 weights come from Ultralytics, which is AGPL-3.0 licensed.
