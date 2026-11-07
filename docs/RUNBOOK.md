# Robot-day runbook: face detection + hand gestures (head camera directly)

Two terminal windows on the Mac:

| window | how | used for |
|---|---|---|
| **Terminal 1 (Mac)** | Terminal, prompt `aayushnair@...  %` | copying files to / from the robot, the analysis |
| **Terminal 2 (Robot)** | second window (Cmd+N): `ssh unitree@192.168.123.164`, password, `1` at the ROS prompt; prompt `unitree@...:~$` | building and running on the robot |

The program (`build/face_gesture`) runs on the robot; its text output appears in Terminal 2, the picture in the
live view (http://192.168.123.164:8090/). Everything it writes stays in `~/aayush_person_track/` on the robot.
It never moves the robot, never uses sudo, and never stops or starts Unitree's services: you stop
`videohub_pc4` by hand at the start of each run (step 5); `master_service` restarts it by itself, and it gets
the camera back when our program exits (a reboot, step 10, resets everything).

## 0. Connect (Mac)

Robot on (wait 1-2 min), USB LAN adapter + cable, System Settings -> Network -> *USB 10/100/1000 LAN* ->
Details -> TCP/IP: Configure IPv4 **Manually**, IP `192.168.123.222`, subnet `255.255.255.0`, router empty.
`ping -c 3 192.168.123.164` must answer.

## 1. Copy the project (Terminal 1)

The models are made on the Mac once (`~/Downloads/benchmark_face/.venv/bin/python scripts/export_face_hand.py`,
already done). Then:
```bash
cd ~/robot-yolo
ssh unitree@192.168.123.164 "mkdir -p ~/aayush_person_track/models"
scp -r src Makefile scripts unitree@192.168.123.164:aayush_person_track/
scp models/yunet_*.onnx models/palm_192.onnx models/handpose_224.onnx unitree@192.168.123.164:aayush_person_track/models/
```
After code changes only the `src Makefile scripts` line is needed, then step 3.

## 2. Build the engines (Terminal 2, once)

```bash
cd ~/aayush_person_track && chmod +x scripts/*.sh
scripts/build_engines.sh          # yunet_640x384, palm_192, handpose_224, then yunet_416x256 and 320x192; fp16
cat logs/engines.md
```
Each must end with `exit 0` and a table row with a time in ms; the first three are the ones the app needs.
If one says FAILED, paste the output. (Optional: `LEAN=1 ...` builds without cuDNN/cuBLAS for less RAM;
`PRECS="fp16 fp32"` adds fp32 engines for comparisons.)

## 3. Build the app (Terminal 2)

```bash
scripts/build.sh                  # ends with "built build/face_gesture" (same as make -j4)
```

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
cd ~/aayush_person_track && sudo pkill -f videohub_pc4; build/face_gesture --print-every 30
```
Defaults: `/dev/video4` at 1280x720 @ 30 (hands need the pixels), `yunet_640x384`, the palm and hand engines.
Look for `driver reports 1280x720 @ 30.0 FPS, format YUYV`, then one line per second
(`target_id`, face box, `n_faces`, `n_hands`, `left_hand`, `right_hand`, timings). **Live view:** open
http://192.168.123.164:8090/ on the Mac: green box = the followed face, orange skeletons = their hands with the
gesture. Stand 1-2 m away, hold up a hand and make gestures; each recognised gesture prints a line such as
`[   12.34 s] target 1, right hand: Thumbs up`. Check that **left/right** match your real hands (if they are
swapped, tell me). Ctrl+C prints the summary. If it says the camera is busy, run `sudo -v` and try again.
Faces only: add `--no-hands`.

## 7. Recorded clip (Terminal 2, optional)

```bash
sudo pkill -f videohub_pc4; scripts/record_clip.sh 60 frames/clip1.h264      # make gestures for 60 s
build/face_gesture --source frames/clip1.h264 --quiet --summary logs/clip1_640.md
build/face_gesture --source frames/clip1.h264 --quiet --summary logs/clip1_320.md --face-engine models/yunet_320x192.fp16.engine
```
The clip is the camera (1280x720 @ 30) encoded by the hardware encoder; replaying it runs every frame as fast
as possible (the program's maximum rate) on identical frames, e.g. to compare face model sizes.

## 8. Logged live runs (Terminal 2)

Each run: 120 s (`DUR=`), saved in its own folder `runs/<NNN>_<TAG>/` (frames.csv, summary, tegrastats.log,
run.txt, console.log with the gesture events, video.mp4 with `VIDEO=`). Write what you did in `NOTE`.
```bash
sudo -v
sudo pkill -f videohub_pc4; TAG=base NOTE="1.5 m, all 15 gestures, both hands" VIDEO=30 scripts/run_live.sh
sudo pkill -f videohub_pc4; TAG=far NOTE="3 m, gestures" scripts/run_live.sh
sudo pkill -f videohub_pc4; TAG=face320 NOTE="smallest face model" ENGINE=models/yunet_320x192.fp16.engine scripts/run_live.sh
sudo pkill -f videohub_pc4; TAG=faces NOTE="faces only" EXTRA="--no-hands" scripts/run_live.sh
sudo pkill -f videohub_pc4; TAG=nv12 NOTE="VIC colour conversion" EXTRA="--format nv12" scripts/run_live.sh
```
Ctrl+C ends a run early and still saves everything. The live view works during these runs too; it costs some
CPU while a browser is open, so keep it either closed in all runs or open in all (the table shows "live view
frames" per run).

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
