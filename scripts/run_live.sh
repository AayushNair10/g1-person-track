#!/usr/bin/env bash
# Phase 5b (on the robot): a live run on the head camera, with everything it measures kept in its own
# folder runs/<NNN>_<TAG>/ for later comparison (scripts/analyze_runs.py).
# Option B (default source /dev/video4): stop videohub_pc4 by hand first (docs/RUNBOOK.md section 7).
#   scripts/run_live.sh                                        120 s, default engine, 640x360 @ 60 FPS
#   TAG=walk NOTE="2 m, walking left/right, lights on" scripts/run_live.sh
#   DUR=60 ENGINE=models/yolo26n_416x256.fp16.engine TAG=416 scripts/run_live.sh
#   VIDEO=30 scripts/run_live.sh                               also save the first 30 s as an annotated MP4
#   EXTRA="--width 848 --height 480" / EXTRA="--format nv12" / EXTRA="--hw-resize"   extra app options
# runs/<NNN>_<TAG>/: frames.csv (every frame), summary.md + summary.md.json (the app's summary), tegrastats.log
# (CPU / GPU / RAM / power once a second), console.log (what the app printed), run.txt (settings, notes),
# video.mp4 (if VIDEO). Nothing outside ~/aayush_person_track is written; nothing runs with sudo.
set -uo pipefail
cd "$(dirname "$0")/.."
DUR=${DUR:-120}
ENGINE=${ENGINE:-models/yolo26n_640x384.fp16.engine}
SOURCE=${SOURCE:-/dev/video4}
TAG=${TAG:-live}
VIDEO=${VIDEO:-0}
NOTE=${NOTE:-}
EXTRA=${EXTRA:-}
APP=build/person_track
[ -x "$APP" ] || { echo "build first: make -j4"; exit 1; }
if [[ "$SOURCE" == /dev/video* ]] && pgrep -f videohub_pc4 >/dev/null; then
    echo "videohub_pc4 is running and holds the camera. Stop it first: sudo pkill -f videohub_pc4"
    exit 1
fi

mkdir -p runs
last=$(ls -d runs/[0-9][0-9][0-9]_* 2>/dev/null | sed 's|^runs/\([0-9]*\)_.*|\1|' | sort -n | tail -1)
RUN=runs/$(printf %03d $((10#${last:-0} + 1)))_$TAG
mkdir -p "$RUN"

video_args=()
[ "$VIDEO" != 0 ] && video_args=(--save-video "$RUN/video.mp4" --save-seconds "$VIDEO")
# shellcheck disable=SC2206
cmd=("$APP" --engine "$ENGINE" --source "$SOURCE" --duration "$DUR" --quiet --label "$(basename "$RUN")"
     --csv "$RUN/frames.csv" --summary "$RUN/summary.md" ${video_args[@]+"${video_args[@]}"} $EXTRA)

{
    echo "run: $(basename "$RUN")"
    echo "note: $NOTE"
    echo "started (robot clock, may be unset): $(date '+%Y-%m-%d %H:%M:%S')"
    echo "command: ${cmd[*]}"
    echo "app build: $(md5sum "$APP" | cut -c1-12)"
    echo "engine: $ENGINE ($(du -k "$ENGINE" 2>/dev/null | cut -f1) KB)"
    echo "host: $(hostname), $(head -1 /etc/nv_tegra_release 2>/dev/null)"
} >"$RUN/run.txt"

echo "tegrastats_start_unix: $(date +%s.%N)" >>"$RUN/run.txt"
tegrastats --interval 1000 >"$RUN/tegrastats.log" 2>&1 &
TS=$!
stop_ts() { { kill "$TS" && wait "$TS"; } 2>/dev/null; }
trap stop_ts EXIT
echo "live run $RUN: ${DUR}s, $ENGINE, source $SOURCE (Ctrl+C stops early and still writes everything)"
# tee -i: Ctrl+C stops the app (which then writes its summary), not the copy of its output
"${cmd[@]}" 2>&1 | tee -i "$RUN/console.log"
stop_ts
echo "finished (robot clock): $(date '+%Y-%m-%d %H:%M:%S')" >>"$RUN/run.txt"
echo
echo "saved in $RUN:"
ls -la "$RUN"
