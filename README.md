# G1 face detection + hand gestures

Real-time face detection and hand-gesture recognition for the person the Unitree G1 is following, on its
onboard Jetson (Orin NX 16 GB, JetPack 5.1.1), from the head camera. **YuNet** finds faces, a C++ **OC-SORT**
keeps their IDs and one face is followed; around that person, Google's **MediaPipe** palm detector and hand
landmark model find their hands, and geometric rules turn the 21 landmarks per hand into **15 gestures**.
Lightweight first: C++ only, everything as TensorRT fp16, nothing installed on the robot (CUDA 11.4, TensorRT
8.5.2, OpenCV 4.2 and GStreamer are already there). Face *recognition* (who it is) can come later on top of
the face points YuNet already gives.

```
head camera, RealSense D435i colour sensor read directly (/dev/video4, 1280x720 @ 30; Unitree's videohub_pc4
  is stopped on the same command line)
  -> camera thread (newest frame wins), frames straight into pinned memory, uploaded to the GPU once
  -> faces: GPU letterbox -> YuNet 640x384 (TensorRT; decoding + best 100 inside) -> NMS -> OC-SORT
            -> followed face (largest at start, ID kept, re-locked by overlap)
  -> hands of the followed person only:
       followed hands:  rotated crop from last frame's landmarks (GPU) -> 21 landmarks (TensorRT)
       fewer than 2:    palm detector on a region around the face (GPU letterbox, TensorRT) -> new hands
       landmarks -> finger states -> 15 gestures -> smoothed (4 of the last 7 frames) -> events
  -> stdout / CSV per frame, gesture events, live view in a browser, annotated MP4, run summary
```

## Status

| | State |
|---|---|
| Models | **done** on the laptop: exported for TensorRT 8.5 and checked (below) |
| C++ app (`build/face_gesture`) | **written and checked on the laptop**: every input tensor, decoder, crop, landmark mapping and gesture rule compared with an independent Python reference; all sources compile against the real TensorRT 8.5 / CUDA 11.4 / OpenCV 4.2 headers |
| On the robot | not run yet: build the engines, build the app, run ([docs/RUNBOOK.md](docs/RUNBOOK.md)) |
| Gesture accuracy on real hands | **open**: needs a clip of you making the gestures (your existing clips have no hands) |
| Person tracking (YOLO26n) | moved, unchanged, to [extras/person_yolo/](extras/person_yolo/README.md) (not built by default) |

## Models

From OpenCV Zoo, downloaded by `scripts/export_face_hand.py` (checksums checked):

| model | job | input | compute | licence |
|---|---|---|---|---|
| YuNet (`face_detection_yunet_2023mar`) | faces: box + 5 points (eyes, nose, mouth corners) | 640x384 (also 416x256, 320x192) | 0.41 GFLOPs (0.18, 0.10) every frame | MIT |
| MediaPipe palm detector (`palm_detection_mediapipe_2023feb`) | palms (box + 7 points) | 192x192, region around the followed face | 0.57 GFLOPs, only while fewer than 2 hands are followed | Apache-2.0 |
| MediaPipe hand landmarks (`handpose_estimation_mediapipe_2023feb`) | 21 points per hand, presence, left/right | 224x224 rotated hand crop | 0.29 GFLOPs per followed hand | Apache-2.0 |

Typical work per frame: 0.4 GFLOPs with no hands, about 1 GFLOP with two hands followed, plus 0.57 on frames
that search for a hand; the previous YOLO26n person detector alone was 3.26.

`scripts/export_face_hand.py` makes each model one-input, one-output and fixed-size for TensorRT 8.5:
YuNet's decoding (grid offsets, exp, score = sqrt(cls x obj)) and the top-100 selection run inside the graph,
so the CPU reads 100 candidates instead of ~80,000 values; the palm detector carries its anchors and keeps
its best 32; the hand model's 4 outputs become one vector. All ops are in the TensorRT 8.5 parser's list.

## Gestures

Open palm, Fist, Thumbs up, Thumbs down, Pointing up / down / left / right, Victory, Three, OK, Pinch, Rock,
I love you, Call me (`src/gestures.cpp`). Each finger is straight, half or curled from its joint angles and
distances in the model's 3-D world landmarks (metres, so the same at any distance or hand rotation); thumb
and index tips together make OK / Pinch; directions (thumbs, pointing) come from the image, as the camera sees
them. A gesture is reported when it is the label of at least 4 of the last 7 frames, and printed as an event
when it changes: `[   12.34 s] target 3, right hand: Thumbs up`. "left/right hand" is the person's own
side; the hand model assumes a mirrored (selfie) image, so for the robot's unmirrored camera its output is
swapped (`--mirrored` if a camera is mirrored).

Only the followed person's hands count: palms are searched in a region around their face (6 face widths
wide, from 1 face height above to 3.5 below, `--region`), and a followed hand that wanders more than 6 face
widths from the face is dropped.

## Live view in a browser

While the program runs, open **http://192.168.123.164:8090/** in a browser on the laptop: the camera picture
with the followed face in green (and its 5 points), other faces in grey, the followed person's hands as orange
skeletons labelled with side and gesture, and a status line (frame, results per second, latency, faces, hands,
target ID). Also `/stream` (the MJPEG stream itself) and `/snapshot.jpg`. POSIX sockets + OpenCV's JPEG encoder,
nothing extra. Frames are handed over (at most
`--view-fps`, default 15) only while a browser is open, and drawn and encoded on the view's own thread, so with
no browser open it costs nothing. `--view-port 0` turns it off; if the port is taken the tracker runs without it.
Checked on the laptop with real sockets (`tests/desktop/check_live_view.sh`: page, stream rate, snapshot,
viewer count, clean shutdown with a browser connected).

## Analytics from the robot

`scripts/run_live.sh` gives every run its own folder on the robot, `~/aayush_person_track/runs/<NNN>_<TAG>/`:

| file | contents |
|---|---|
| `frames.csv` | every frame: time, followed face ID and box, faces and hands in view, each hand's gesture, time of each stage, latency |
| `summary.md`, `summary.md.json` | the run's totals: results per second, latency percentiles, followed-face ID changes, frames with hands, gestures recognised, memory |
| `tegrastats.log` | whole robot once a second: CPU per core, GPU, RAM, power, memory bus |
| `run.txt` | settings: command, engine, camera mode, your `NOTE`, app build, start times |
| `console.log` | what the app printed, including every gesture event |
| `video.mp4` | annotated video (only with `VIDEO=<seconds>`) |

On the laptop, `scripts/get_runs.sh` copies all of them into `results/runs/` (not in git: the repository is
public and the videos show people) and runs `scripts/analyze_runs.py`, which writes `comparison.md` /
`comparison.csv` (one row per run) and `comparison.png` (runs side by side), plus `analysis.png` in each run
folder (results per second, latency, target presence and ID changes, CPU/GPU load and power over time). The
comparison table also shows, per run, the share of frames with a hand and how often each gesture was recognised.

## What was checked on the laptop

**Models** (`scripts/export_face_hand.py`, `models/face_hand_report.json`): each exported graph gives the
original model's numbers (YuNet decoded: max difference 6e-5 px; palm and hand: identical), YuNet's decoding
matches OpenCV's own `FaceDetectorYN` on frames of your clips (16 of 16 faces, box IoU 0.9999996), the palm
anchors match OpenCV Zoo's table, and every graph passes the TensorRT 8.5 op check (`scripts/trt85_check.py`).

**C++ vs an independent Python reference** (`tests/desktop/check_facehand.py`, `scripts/facehand_ref.py`): face
input tensors bit-exact with OpenCV (640x384, 320x192; 416x256 within one 8-bit level), palm-region tensors
within one level, rotated hand crops within 0.005 levels, for BGR and NV12 frames; face decoding (5e-7 px) and
palm decoding + hand crop (1e-4) on real model outputs; hand landmarks to the frame, next crop, finger states
and gestures on 40 real hand-model outputs: identical. Gesture rules on synthetic 3-D hands: all 15 gestures
recognised, each turned 3 ways and at 2 sizes (90 of 90). On your webcam clip the pipeline finds the face in 142
of 150 frames and, correctly, no hands.

**Not checked yet:** gesture accuracy on real hands, the left/right swap, and everything on the robot.

**Also:** `tests/desktop/syntax_check.sh` (all sources against the real headers, no warnings),
`tests/desktop/check_live_view.sh` (live view over real sockets).

## Running

On the robot (details and the order: [docs/RUNBOOK.md](docs/RUNBOOK.md)):

```bash
sudo pkill -f videohub_pc4; build/face_gesture --print-every 30          # live view: http://192.168.123.164:8090/
sudo pkill -f videohub_pc4; TAG=walk NOTE="2 m, gestures" scripts/run_live.sh   # logged run -> runs/<NNN>_walk/
build/face_gesture --face-engine models/yunet_320x192.fp16.engine         # smallest face model
build/face_gesture --no-hands                                             # faces only
build/face_gesture --help
```
(`pkill` and the program on one line, no pause: Unitree's `master_service` restarts `videohub_pc4` within
~2-3 s, and the program opens the camera before loading its engines.)

Per-frame output: `frame,timestamp,target_id,x1,y1,x2,y2,conf,n_faces,n_hands,left_hand,right_hand,wait_ms,
upload_ms,pre_ms,infer_ms,hand_ms,post_ms,track_ms,total_ms,event`; boxes in camera pixels (1280x720);
`left_hand` / `right_hand` hold that hand's gesture (`-` = hand seen, no gesture); `target_id` -1 = nobody.

On the laptop, the same pipeline on a video file (ONNX Runtime), e.g. for tuning the gesture rules:
```bash
~/Downloads/benchmark_face/.venv/bin/python scripts/facehand_ref.py ~/Downloads/benchmark_face/gestures.mov --out /tmp/g.mp4
```

## What runs

**On the robot** (all preinstalled with JetPack 5.1.1, nothing added): TensorRT 8.5.2 (`libnvinfer`; `trtexec`
to build engines), CUDA 11.4 (`libcudart`; `nvcc` for the two input kernels), OpenCV 4.2 (`core`, `imgproc`,
`videoio`, `imgcodecs`: camera, MP4 writer, drawing, JPEG for the live view), GStreamer 1.16 (only for `.h264`
files and `--format nv12`), g++ 9.4, the C++17 standard library and pthreads; `tegrastats` and Python 3.8's
standard library for the measurement scripts.

**On the laptop** (the existing venv, nothing installed): onnx, onnxruntime, numpy, opencv-python, matplotlib
(charts); the three models are downloaded from OpenCV Zoo.

## Layout

```
src/                 C++ app (build/face_gesture)
  main.cpp           loop, options, CSV, gesture events, summary (.md + .json)
  camera.*           V4L2 / GStreamer / file reader thread into pinned memory
  preprocess.cu/.hpp, preprocess_math.hpp   GPU letterbox (region) and rotated crop (+ the shared per-pixel code)
  trt_engine.*       TensorRT 8.5 engine (bindings, enqueueV2, CUDA graph; NCHW or NHWC input)
  face_detector.hpp  YuNet output -> faces (NMS, back to frame pixels)
  hands.hpp          palm output -> palms, hand crops, landmarks -> frame, next crop, search region
  gestures.*         finger states, 15 gesture rules, smoothing
  ocsort.*           OC-SORT (port of Ultralytics 8.4; validated in extras/person_yolo)
  target.hpp         followed-face choice: largest at start, keep ID, IoU re-lock
  video_sink.*       annotated MP4 writer thread + drawing (faces, points, hand skeletons, labels)
  live_view.*        live view in a browser (MJPEG over HTTP, port 8090)
Makefile             make -j4 -> build/face_gesture (or scripts/build.sh without make)
scripts/             export_face_hand.py, trt85_check.py, facehand_ref.py (laptop); deploy.sh, get_runs.sh,
                     fetch_results.sh, g1_env.sh, phase1_verify.sh (laptop -> robot); build.sh, build_engines.sh,
                     record_clip.sh, run_live.sh, make_results.py (robot); analyze_runs.py (both)
tests/desktop/       laptop checks: C++ vs Python reference, live view, compile check
models/              ONNX files (generated, not in git), face_hand_report.json
extras/person_yolo/  the YOLO26n person tracker, self-contained (own src/, scripts/, tests/, models/)
docs/RUNBOOK.md      robot-day steps
```

## Reproducing the laptop checks

```bash
PY=~/Downloads/benchmark_face/.venv/bin/python
CLIPS="$HOME/Downloads/benchmark_face/bench_results/webcam_clip.mp4 $HOME/Downloads/benchmark_face/tilt_results/tilt.mp4"
$PY scripts/export_face_hand.py                   # models/*.onnx + face_hand_report.json
mkdir -p build/desktop
clang++ -std=c++17 -O2 tests/desktop/test_facehand.cpp src/gestures.cpp -o build/desktop/test_facehand
$PY tests/desktop/check_facehand.py --clips $CLIPS
```

## Notes

- Licences: YuNet MIT; MediaPipe palm / hand models Apache-2.0; the OC-SORT port comes from Ultralytics
  (AGPL-3.0), so keep that in mind when choosing a licence for this repository.
- Everything in `src/` was written for this repository; the lab's FaceHandTrack was not used.
