# Robot-day runbook (Option B: the head camera directly)

Two terminal windows on the Mac:

| window | how | used for |
|---|---|---|
| **Terminal 1 (Mac)** | Terminal, prompt `aayushnair@...  %` | copying files to / from the robot, the analysis |
| **Terminal 2 (Robot)** | second window (Cmd+N): `ssh unitree@192.168.123.164`, password, `1` at the ROS prompt; prompt `unitree@...:~$` | building and running on the robot |

The program runs on the robot; its text output appears in Terminal 2. Everything it writes stays in
`~/aayush_person_track/` on the robot. It never moves the robot, never uses sudo, and never stops or starts
Unitree's services: you stop `videohub_pc4` by hand in step 5 and a reboot brings it back (step 10).

## 0. Connect (Mac)

Robot on (wait 1-2 min), USB LAN adapter + cable, System Settings -> Network -> *USB 10/100/1000 LAN* ->
Details -> TCP/IP: Configure IPv4 **Manually**, IP `192.168.123.222`, subnet `255.255.255.0`, router empty.
`ping -c 3 192.168.123.164` must answer.

## 1. Copy the project (Terminal 1)

```bash
cd ~/robot-yolo
ssh unitree@192.168.123.164 "mkdir -p ~/aayush_person_track/models"
scp -r src Makefile scripts unitree@192.168.123.164:aayush_person_track/
scp models/*.onnx unitree@192.168.123.164:aayush_person_track/models/      # once; ~100 MB
```
After code changes only the `src Makefile scripts` line is needed, then `make -j4` (step 3).

## 2. Build the engines (Terminal 2, once)

```bash
cd ~/aayush_person_track && chmod +x scripts/*.sh
PRECS=fp16 scripts/build_engines.sh models/yolo26n_640x384.onnx          # the default engine, 2-5 min
```
Must end with `exit 0` and a table row with a time in ms. If it says FAILED: build
`models/yolo26n-raw_640x384.onnx` (same weights, raw head) or `models/yolo11n_640x384.onnx` and pass that engine
with `ENGINE=...` / `--engine`. Optional, for the compute comparison (step 8):
```bash
PRECS=fp16 scripts/build_engines.sh models/yolo26n_416x256.onnx models/yolo26n_320x192.onnx
LEAN=1 PRECS=fp16 scripts/build_engines.sh models/yolo26n_640x384.onnx models/yolo26n_416x256.onnx
DLA=0 scripts/build_engines.sh models/yolo26n_640x384.onnx
cat logs/engines.md
```
`lean` = no cuDNN/cuBLAS (less RAM, faster start); `dla0` = runs on the Orin's deep-learning accelerator and
leaves the GPU free (layers it cannot run fall back to the GPU). Skip any that fail.

## 3. Build the app (Terminal 2)

```bash
make -j4                                     # ends with "built build/person_track"; clock-skew warnings are fine
```
Same without make (FaceHandTrack-style, same preinstalled tools): `scripts/build.sh`.

## 4. Check the camera (Terminal 2, service still running)

```bash
cat /sys/class/video4linux/video4/name                  # RealSense ... RGB
v4l2-ctl -d /dev/video4 --list-formats-ext | head -60   # modes (skip if "command not found")
ls -l /dev/video4; groups                                # you must be in the device's group (usually video)
```
Not in the group: stop here (the app must not run with sudo).

## 5. How to start anything that uses the camera (Terminal 2)

`videohub_pc4` holds `/dev/video4`, and on this robot `master_service` restarts it **about 2-3 s after it is
stopped** (checked on the robot). So stop it and start our program **on the same line, with no pause**; the
program opens the camera before it loads the engine, so it holds the camera first. While our program runs,
`master_service` keeps restarting `videohub_pc4`, which cannot get the camera; when our program exits,
`videohub_pc4` takes it back, so **every run starts with the pkill again**. Unitree's own camera feed (its app,
the SDK's VideoClient) is unavailable during our runs: tell anyone else using the robot. Do not stop
`master_service` (it runs Unitree's other services too).

```bash
sudo -v        # type the password now, so the next sudo does not pause (remembered ~15 min)
```
`sudo pkill -f videohub_pc4` may print "Terminated": `pkill -f` also matches the `sudo` command itself. Use `;`
(not `&&`) after it.

## 6. Quick test (Terminal 2)

```bash
cd ~/aayush_person_track && sudo pkill -f videohub_pc4; build/person_track --print-every 30
```
Defaults: `/dev/video4`, 640x360 @ 30 FPS (exactly the picture inside the 640x384 model input), the default
engine. Look for `driver reports 640x360 @ 30.0 FPS, format YUYV`. One line per second:
`frame,timestamp,target_id,x1,y1,x2,y2,conf,n_persons,...` (`target_id` -1 = nobody; box in 640x360 pixels).
Ctrl+C prints the summary (results per second should be close to 30). If it says the camera is busy, run
`sudo -v` and try again (the restart won the race). Other mode if 640x360 is not offered:
`--width 640 --height 480`. Below 30 FPS in a dim room = the camera's auto exposure: more light helps.

## 7. Recorded clip and offline checks (Terminal 2, optional)

```bash
sudo pkill -f videohub_pc4; scripts/record_clip.sh 60 frames/clip1.h264   # walk around for 60 s
scripts/run_offline.sh frames/clip1.h264              # fp16 vs fp32, sizes, variants on identical frames
python3 scripts/make_results.py                       # -> logs/RESULTS.md
```
The clip is the camera (640x360 @ 30) encoded by the hardware encoder; the app and the recorder cannot use the
camera at the same time.

## 8. Logged live runs (Terminal 2)

Each run: 120 s (`DUR=`), saved in its own folder `runs/<NNN>_<TAG>/` (frames.csv, summary, tegrastats.log,
run.txt, console.log, video.mp4 with `VIDEO=`). Write what you did in `NOTE` so you can tell runs apart later.
During each run: stand ~2 m away, walk left/right, turn around, crouch, leave the view ~5 s and come back, and if
possible have a second person cross in front of you.
```bash
sudo -v
sudo pkill -f videohub_pc4; TAG=base NOTE="640x360@30, default engine, walking" VIDEO=30 scripts/run_live.sh
sudo pkill -f videohub_pc4; TAG=nv12 NOTE="VIC colour conversion" EXTRA="--format nv12" scripts/run_live.sh
sudo pkill -f videohub_pc4; TAG=416 NOTE="416x256 engine" ENGINE=models/yolo26n_416x256.fp16.engine scripts/run_live.sh
sudo pkill -f videohub_pc4; TAG=lean416 ENGINE=models/yolo26n_416x256.fp16.lean.engine scripts/run_live.sh
sudo pkill -f videohub_pc4; TAG=dla ENGINE=models/yolo26n_640x384.fp16.dla0.engine scripts/run_live.sh
```
Ctrl+C ends a run early and still saves everything.

## 9. Copy and compare (Terminal 1)

```bash
cd ~/robot-yolo
scripts/get_runs.sh                         # robot runs/ -> results/runs/, logs/ -> results/logs/, then the analysis
open results/runs/comparison.png            # all runs side by side
open results/runs/001_base/analysis.png     # one run over time
open results/runs/001_base/video.mp4
```
`results/runs/comparison.md` / `.csv` have the numbers (one row per run). `results/` stays out of git (public
repository; the videos show people). The same tables without charts on the robot:
`python3 scripts/analyze_runs.py runs`.

## 10. Restore (Terminal 2)

```bash
sudo reboot
```
Terminal 2 disconnects; `videohub_pc4` starts again with the reboot.

## Not without your explicit OK

- Anything with sudo beyond steps 4-8 (`pkill -f videohub_pc4`, `fuser`) and 10, stopping `master_service`, power mode, files outside `~/aayush_person_track`.
- Phase 6 (ROS 2 topic, head location with YOLO26n-pose).
