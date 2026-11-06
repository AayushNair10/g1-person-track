# Robot-day runbook

Everything that does not need the robot is done (see the README). This is the order for the robot session.
Steps marked **writes** change files on the robot, always only inside `/home/unitree/aayush_person_track/`,
and are announced before they run. Nothing is installed on the robot, sudo is only used for the two
read-only Phase 1 checks (`nvpmodel -q`, `fuser -v`), and `videohub_pc4`, power mode and system files are
never touched.

## 0. Connect (you)

1. Power on the robot, wait 1-2 min.
2. USB LAN adapter on the Mac, cable to the robot.
3. System Settings -> Network -> *USB 10/100/1000 LAN* -> Details -> TCP/IP: Configure IPv4 **Manually**,
   IP `192.168.123.222`, subnet `255.255.255.0`, router empty.
4. In your own terminal, open one shared SSH connection (password once; `-N` opens no shell, so no ROS prompt):

   ```bash
   ssh -fNM -o ControlPath=~/.ssh/cm-g1 -o ControlPersist=12h unitree@192.168.123.164
   ```
   Close it at the end with `ssh -O exit -o ControlPath=~/.ssh/cm-g1 unitree@192.168.123.164`.

## 1. Verify the robot (read-only)

```bash
ping -c 3 192.168.123.164
source scripts/g1_env.sh && g1_check
mkdir -p results && ssh -n -o ControlPath=~/.ssh/cm-g1 unitree@192.168.123.164 "$(cat scripts/phase1_verify.sh)" | tee results/phase1.txt
```
Compare with the lab guide: Orin NX 16 GB, L4T R35.3.1, TensorRT 8.5.2, OpenCV 4.2.0, MAXN, `videohub_pc4`
holding `/dev/video4`, `nvv4l2decoder` present, free disk > 1 GB. If `sudo -n` is refused (password
needed), run these two yourself in an interactive `ssh unitree@192.168.123.164`: `sudo nvpmodel -q` and
`sudo fuser -v /dev/video*`. **Stop if anything differs.**

## 2. Copy the project (writes)

```bash
scripts/deploy.sh all      # mkdir ~/aayush_person_track/{models,src,build,logs,scripts,frames}; copy src, Makefile, scripts, 10 ONNX (~100 MB)
```

## 3. Build the engines on the robot (writes models/*.engine, logs/trtexec_*)

```bash
g1_ssh "cd ~/aayush_person_track && nohup scripts/build_engines.sh > logs/build_engines.out 2>&1 &"
g1_ssh "tail -5 ~/aayush_person_track/logs/build_engines.out"      # progress; ~1-3 min per engine, 12 engines
g1_ssh "cat ~/aayush_person_track/logs/engines.md"                  # table at the end
```
Order: 640x384 first (the default engine), then 640x640, 416x416, 416x256, 320x320, 320x192, each fp16 + fp32.

If YOLO26n fails (the NMS-free head ends in TopK / GatherElements / an integer Mod; TensorRT 8.5's parser
supports all of them, but this is the part most likely to fail):
1. same weights, raw head + NMS in C++: `scripts/build_engines.sh models/yolo26n-raw_640x384.onnx`
2. YOLO11n: `scripts/build_engines.sh models/yolo11n_640x384.onnx`
3. the lab's `make_trt85_onnx.py` approach (fold shapes, split ops) on the laptop, then copy again.
The app reads the output shape and picks the right decoder by itself.

### 3b. Lower-compute engine variants (writes models/*.engine, ~2 min each)

```bash
g1_ssh "cd ~/aayush_person_track && LEAN=1 PRECS=fp16 scripts/build_engines.sh models/yolo26n_640x384.onnx models/yolo26n_416x256.onnx models/yolo26n_320x192.onnx"
g1_ssh "cd ~/aayush_person_track && DLA=0 scripts/build_engines.sh models/yolo26n_640x384.onnx models/yolo26n_416x256.onnx"
g1_ssh "cat ~/aayush_person_track/logs/engines.md"
```
`lean` = built without cuDNN/cuBLAS, so TensorRT never loads those libraries (expect a few hundred MB less RAM
and a faster start; GPU time should stay about the same). `dla0` = runs on DLA core 0 (the Orin NX's separate
deep-learning accelerator), so the GPU is left free; layers the DLA cannot run fall back to the GPU (the build
prints how many). The DLA is usually slower than the GPU, which is fine while it stays well under the 66 ms between
frames. If a DLA build fails, skip it.

## 4. Build the app (writes build/)

```bash
g1_ssh "cd ~/aayush_person_track && make -j4 2>&1 | tail -5"      # clock-skew warnings are expected
g1_ssh "cd ~/aayush_person_track && build/person_track --source videohub --duration 10 --print-every 15"
```
Smoke test: the camera line should say `1280x720 BGR`, and rows should show a target ID and a box when
someone is in view.

## 5a. Offline checks on a recorded clip (writes frames/, logs/)

Walk around in front of the head camera for 60 s while this records (only listens to the multicast stream):
```bash
g1_ssh "cd ~/aayush_person_track && scripts/record_clip.sh 60 frames/clip1.h264"
g1_ssh "cd ~/aayush_person_track && scripts/run_offline.sh frames/clip1.h264"
```
fp16 vs fp32 agreement (same input tensors) and the pipeline's maximum frame rate without the camera
limit, for all six sizes. Then the compute savings on the same clip (`--hw-resize`, `--every 2`, and the lean /
DLA engines if built), each compared with the 640x384 baseline: `logs/agreement.md` shows, frame by frame,
whether each variant still reports the same person in the same place.

## 5b. Live, 2 minutes, Unitree services running (writes logs/)

During each run: stand ~2 m away, walk left/right, turn around, crouch, leave the view for ~5 s and come
back, and if possible have a second person cross in front of you.
```bash
g1_ssh "cd ~/aayush_person_track && TAG=lab scripts/run_live.sh"                               # lab pipeline (BGR)
g1_ssh "cd ~/aayush_person_track && SOURCE=videohub-nv12 TAG=nv12 scripts/run_live.sh"          # NV12, no CPU colour conversion
g1_ssh "cd ~/aayush_person_track && SOURCE=videohub-nv12 TAG=nv12_dec EXTRA='--dec-opts enable-max-performance=1' scripts/run_live.sh"
g1_ssh "cd ~/aayush_person_track && ENGINE=models/yolo26n_640x640.fp16.engine TAG=640sq scripts/run_live.sh"  # your original 640x640
g1_ssh "cd ~/aayush_person_track && DUR=30 VIDEO=30 TAG=video scripts/run_live.sh"             # annotated MP4
```
Optional A/B for latency: `EXTRA=--zero-copy`, `EXTRA=--no-graph`, `EXTRA='--dec-opts disable-dpb=1'`
(only if the stream has no B-frames; if the picture breaks, drop it).

## 5c. Lowest compute, live (writes logs/; 60 s each, same movements)

```bash
g1_ssh "cd ~/aayush_person_track && DUR=60 SOURCE=videohub-nv12 TAG=hw EXTRA=--hw-resize scripts/run_live.sh"
g1_ssh "cd ~/aayush_person_track && DUR=60 SOURCE=videohub-nv12 ENGINE=models/yolo26n_640x384.fp16.lean.engine TAG=hw_lean EXTRA=--hw-resize scripts/run_live.sh"
g1_ssh "cd ~/aayush_person_track && DUR=60 SOURCE=videohub-nv12 ENGINE=models/yolo26n_416x256.fp16.lean.engine TAG=hw_lean_416 EXTRA=--hw-resize scripts/run_live.sh"
g1_ssh "cd ~/aayush_person_track && DUR=60 SOURCE=videohub-nv12 ENGINE=models/yolo26n_640x384.fp16.dla0.engine TAG=hw_dla EXTRA=--hw-resize scripts/run_live.sh"
```
Compare GPU %, CPU %, RAM and power (tegrastats table) and the target columns against the `lab` run. Pick the
smallest input whose `logs/agreement.md` row stays near 100% "same person & place" on the clip; `--every 2`
halves everything again if 7.5 results per second are enough for the robot.
`--hw-resize` makes the VIC scale 1280x720 to the model's picture size (640x360 for 640-wide inputs) before the
frame leaves GStreamer: the CPU and the copy to the GPU handle a quarter of the pixels. It assumes the stream
is 1280x720 (step 4 shows the real size; otherwise pass `--width/--height`).

## 6. Results back to the laptop

```bash
scripts/fetch_results.sh          # logs -> results/robot_<time>/, RESULTS.md -> docs/RESULTS.md (+ copy on the robot)
```
Then commit `docs/RESULTS.md` (raw CSV / logs / MP4 stay out of git).

## 7. Option B: the camera directly (you stop the service by hand)

Unitree's camera feed (its app, the SDK's VideoClient) stops until the robot is rebooted: tell anyone else
using the robot. This program never stops or restarts the service.

```bash
cat /sys/class/video4linux/video4/name                 # should be the D435i's RGB node
v4l2-ctl -d /dev/video4 --list-formats-ext | head -60  # modes (if v4l2-ctl is installed)
sudo pkill -f videohub_pc4
sleep 5; pgrep -a videohub                             # nothing = stopped (if it is back: master_service, ask the lab)
sudo fuser -v /dev/video*                              # nothing listed for /dev/video4
cd ~/aayush_person_track
build/person_track --source /dev/video4 --width 640 --height 360 --fps 60 --print-every 60
```
640x360 is exactly the picture inside the 640x384 model input, so nothing is resized. Variants:
`--format nv12` (the VIC converts YUYV, no CPU colour conversion), `--width 848 --height 480` (+ `--hw-resize`),
`--width 1280 --height 720 --fps 30 --hw-resize`. Logged runs: `SOURCE=/dev/video4 TAG=optB
EXTRA="--width 640 --height 360 --fps 60" scripts/run_live.sh`. Boxes are in the camera's own pixels
(640x360 here). Below 60 FPS in a dim room: the camera's auto exposure is lengthening the exposure (more light
helps). Bring the service back with a reboot.

## Not without your explicit OK

- Anything with sudo beyond Phase 1 and step 7, reboots, power mode, files outside `~/aayush_person_track`.
- Phase 6 (ROS 2 topic, head location with YOLO26n-pose).
