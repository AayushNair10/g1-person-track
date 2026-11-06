# g1-person-track

Real-time single-person tracking on the Unitree G1's onboard Jetson (Orin NX 16 GB, JetPack 5.1.1) using the
head camera: **YOLO26n** (TensorRT fp16) finds people, a C++ **OC-SORT** keeps stable IDs, and one target person
is followed. Its box and ID are printed every frame. Lightweight first: C++ only, nothing installed on the robot
(CUDA 11.4, TensorRT 8.5.2, OpenCV 4.2 and GStreamer are already there).

```
head camera (videohub_pc4, H.264 multicast 1280x720 @ 15 FPS)
  -> GStreamer + hardware decoder (camera thread, newest frame wins, decoded straight into pinned memory)
  -> one CUDA kernel: letterbox 1280x720 -> 640x384, BGR/NV12 -> RGB, /255, CHW, into the TensorRT input
  -> TensorRT YOLO26n fp16 (enqueueV2, replayed as a CUDA graph)
  -> decode person boxes (NMS-free head: no NMS) -> C++ OC-SORT -> target selector
  -> stdout / CSV: frame, timestamp, target_id, x1, y1, x2, y2, conf + per-stage timings (+ annotated MP4)
```

## Status

| Phase | State |
|---|---|
| 1 Verify robot | script ready (`scripts/phase1_verify.sh`), runs on robot day |
| 2 Export + check ONNX | **done** on the laptop, all checks pass (below) |
| 3 Build engines | script ready (`scripts/build_engines.sh`), runs on the robot |
| 4 C++ app | **written and checked on the laptop**: tracker, preprocessing and decoding tested against Ultralytics; all sources compile against the real TensorRT 8.5 / CUDA 11.4 / OpenCV 4.2 headers. Compiled on the robot. |
| 5 Validate on robot | scripts ready (`run_offline.sh`, `run_live.sh`, `make_results.py`) |
| 6 ROS 2 / pose | only on request |

Robot-day steps: [docs/RUNBOOK.md](docs/RUNBOOK.md). Results will land in `docs/RESULTS.md`.

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
`tests/desktop/check_tracker.py` feeds identical detections to both on your two clips (2 heads x 2 clips x 30/15 FPS:
8 runs, 7,350 frames): **0 frames differ, boxes agree to 1e-4 px**. Over the benchmark's window (frames 15-315 of
`webcam_clip.mp4`) the target keeps one ID with 93% coverage, as in the benchmark (0 switches, 92.7%).

**Preprocessing and decoding.** The CUDA kernel's per-pixel code (`src/preprocess_math.hpp`) is shared
with a CPU build. `tests/desktop/check_pipeline.py` shows it is **bit-exact** against OpenCV's
platform-independent bilinear (`INTER_LINEAR_EXACT`) at 640x384 / 640x640 / 320x320 / 320x192, for BGR and
for NV12 frames, and within 1 grey level at 416 sizes. Decoding (both heads, with NMS) is identical to the
Python reference. End-to-end detections match with IoU >= 0.999.

**Compilation.** `tests/desktop/syntax_check.sh` compiles every source (and both sides of the CUDA file)
against the real TensorRT 8.5, CUDA 11.4 and OpenCV 4.2.0 headers with `-Wall -Wextra -Wshadow`: no warnings.

## Frame rate

The camera stream is 15 FPS, so live output cannot go above 15 FPS with Option A. What can be optimised is
keeping every frame and cutting the delay from frame arrival to result. Done:

- 16:9 model input (640x384) instead of 640x640: same picture, ~40% less compute
- one CUDA kernel for the whole preprocessing, writing straight into the TensorRT input
- frames decoded directly into pinned memory (no extra CPU copies), one GPU wait per frame
- TensorRT inference recorded once as a CUDA graph and replayed
- camera on its own thread, newest frame wins; video encoding on its own thread
- `videohub-nv12` source: NV12 straight from the hardware decoder, colour conversion done on the GPU, skipping
  the CPU `videoconvert` in the lab pipeline (bit-exact with OpenCV's NV12->BGR + resize)
- switches to measure on the robot: `--zero-copy`, `--no-graph`, `--dec-opts "enable-max-performance=1"`

`scripts/run_offline.sh` measures the pipeline's maximum rate without the camera limit. Going above 15 FPS live
needs Option B (direct RealSense, up to 60 FPS). The app supports it (`--source /dev/video4`), but it means
stopping `videohub_pc4`, so it is never done without your OK.

## Layout

```
src/                 C++ app (person_track)
  main.cpp           loop, options, CSV, timing statistics, summary (.md + .json)
  camera.*           GStreamer / file / V4L2 reader thread into pinned memory
  preprocess.cu/.hpp, preprocess_math.hpp   letterbox kernel (+ the shared per-pixel code)
  trt_engine.*       TensorRT 8.5 engine (bindings, enqueueV2, CUDA graph)
  postprocess.hpp    NMS-free / raw-head decoding, NMS, box mapping
  ocsort.*           OC-SORT (port of Ultralytics 8.4)
  target.hpp         target choice: largest at start, keep ID, IoU re-lock
  video_sink.*       annotated MP4 writer thread
Makefile             build on the robot: make -j4 -> build/person_track
scripts/             export_models.py, validate_onnx.py, yolo_ref.py (laptop); phase1_verify.sh, deploy.sh,
                     fetch_results.sh, g1_env.sh (laptop -> robot); build_engines.sh, record_clip.sh,
                     run_offline.sh, run_live.sh, make_results.py (robot)
tests/desktop/       laptop checks: tracker vs Ultralytics, preprocessing/decoding vs OpenCV/ORT, compile check
models/              ONNX files (generated, not in git), onnx_report.json, onnx_validation.md
docs/                RUNBOOK.md (robot day), RESULTS.md (after the robot runs)
```

## Running

On the robot (after `scripts/deploy.sh all`, `scripts/build_engines.sh`, `make -j4`):

```bash
build/person_track                                   # default engine, head camera via the lab pipeline
build/person_track --source videohub-nv12 --quiet --csv logs/run.csv --summary logs/run.md
build/person_track --source frames/clip1.h264 --compare models/yolo26n_640x384.fp32.engine --quiet
build/person_track --save-video logs/demo.mp4 --save-seconds 30 --duration 30
build/person_track --help
```

Per-frame output: `frame,timestamp,target_id,x1,y1,x2,y2,conf,n_persons,wait_ms,upload_ms,pre_ms,infer_ms,post_ms,track_ms,total_ms,event`.
Boxes are in camera pixels (1280x720). `timestamp` is seconds since start on the monotonic clock (the robot's
wall clock may be unset). `total_ms` is the delay from the frame leaving the decoder to the result.
`target_id` is -1 when no target is visible.

## Reproducing the laptop checks

```bash
PY=~/Downloads/benchmark_face/.venv/bin/python
CLIPS="$HOME/Downloads/benchmark_face/tilt_results/tilt.mp4 $HOME/Downloads/benchmark_face/bench_results/webcam_clip.mp4"
$PY scripts/export_models.py                      # models/*.onnx + onnx_report.json
$PY scripts/validate_onnx.py --clips $CLIPS       # models/onnx_validation.md
mkdir -p build/desktop
clang++ -std=c++17 -O2 tests/desktop/test_tracker.cpp src/ocsort.cpp -o build/desktop/test_tracker
clang++ -std=c++17 -O2 tests/desktop/test_pipeline.cpp -o build/desktop/test_pipeline
$PY tests/desktop/check_tracker.py --clips $CLIPS --models models/yolo26n_640x384.onnx models/yolo26n-raw_640x384.onnx
$PY tests/desktop/check_pipeline.py --clips $CLIPS
```

## Notes

- The OC-SORT port and the YOLO26/YOLO11 weights come from Ultralytics, which is AGPL-3.0 licensed; keep that
  in mind when choosing a license for this repository.
- The lab's FaceHandTrack Jetson example (github.com/kinetrix/Track) was the model for the camera pipeline, the
  TensorRT wrapper and the CUDA-graph replay.
