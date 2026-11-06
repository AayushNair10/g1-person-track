#!/usr/bin/env bash
# Phase 5b (on the robot): live run on the head camera with Unitree's services running, tegrastats
# logged in parallel (read-only; run without sudo).
#   scripts/run_live.sh                                   120 s, default engine, videohub (lab pipeline)
#   DUR=120 ENGINE=models/yolo26n_640x640.fp16.engine SOURCE=videohub-nv12 TAG=nv12 scripts/run_live.sh
#   VIDEO=30 scripts/run_live.sh                          also save the first 30 s as an annotated MP4
#   EXTRA="--dec-opts enable-max-performance=1" ...       extra app options
# Outputs: logs/live_<TAG>.csv, logs/live_<TAG>.md(.json), logs/tegrastats_<TAG>.log [, logs/live_<TAG>.mp4]
set -uo pipefail
cd "$(dirname "$0")/.."
DUR=${DUR:-120}
ENGINE=${ENGINE:-models/yolo26n_640x384.fp16.engine}
SOURCE=${SOURCE:-videohub}
TAG=${TAG:-$(basename "$ENGINE" .engine)_$SOURCE}
VIDEO=${VIDEO:-0}
EXTRA=${EXTRA:-}
APP=build/person_track
[ -x "$APP" ] || { echo "build first: make -j4"; exit 1; }

video_args=()
[ "$VIDEO" != 0 ] && video_args=(--save-video "logs/live_$TAG.mp4" --save-seconds "$VIDEO")

tegrastats --interval 1000 >"logs/tegrastats_$TAG.log" 2>&1 &
TS=$!
trap 'kill $TS 2>/dev/null' EXIT
echo "live run '$TAG': ${DUR}s, $ENGINE, source $SOURCE (Ctrl+C stops early and still writes the summary)"
# shellcheck disable=SC2086
"$APP" --engine "$ENGINE" --source "$SOURCE" --duration "$DUR" --quiet --label "live $TAG" \
    --csv "logs/live_$TAG.csv" --summary "logs/live_$TAG.md" "${video_args[@]}" $EXTRA
kill $TS 2>/dev/null
echo "tegrastats: $(wc -l <"logs/tegrastats_$TAG.log") samples in logs/tegrastats_$TAG.log"
