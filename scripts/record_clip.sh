#!/usr/bin/env bash
# On the robot: record the head camera (Option B: /dev/video4 directly, videohub_pc4 stopped by hand) to an
# H.264 file with the Jetson's hardware encoder, for offline tests on identical frames (run_offline.sh).
#   scripts/record_clip.sh 60 frames/clip_walk.h264            60 s at 640x360 @ 60 FPS
#   W=1280 H=720 FPS=30 scripts/record_clip.sh 60 frames/clip_720p.h264
# Play back with: build/person_track --source frames/clip_walk.h264 ...  (the app cannot run at the same time:
# the camera serves one program)
set -euo pipefail
cd "$(dirname "$0")/.."
SECS=${1:-60}
OUT=${2:-frames/clip_$(date +%H%M%S).h264}
DEV=${DEV:-/dev/video4}
W=${W:-640}
H=${H:-360}
FPS=${FPS:-60}
BITRATE=${BITRATE:-8000000}
if pgrep -f videohub_pc4 >/dev/null; then
    echo "videohub_pc4 is running and holds the camera. Stop it first: sudo pkill -f videohub_pc4"
    exit 1
fi
mkdir -p "$(dirname "$OUT")"
echo "recording ${SECS}s of $DEV (${W}x${H} @ ${FPS}) to $OUT (walk around in front of the robot)..."
timeout -s INT "$SECS" gst-launch-1.0 -q -e \
    v4l2src device="$DEV" ! "video/x-raw,format=YUY2,width=$W,height=$H,framerate=$FPS/1" ! \
    nvvidconv ! "video/x-raw(memory:NVMM),format=NV12" ! \
    nvv4l2h264enc bitrate="$BITRATE" insert-sps-pps=true idrinterval="$FPS" ! \
    h264parse ! video/x-h264,stream-format=byte-stream ! filesink location="$OUT" || true
ls -la "$OUT"
