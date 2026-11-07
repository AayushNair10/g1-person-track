#!/usr/bin/env bash
# On the robot: record the head camera (Option B: /dev/video4 directly) to an
# H.264 file with the Jetson's hardware encoder, for offline tests on identical frames (run_offline.sh).
#   sudo pkill -f videohub_pc4; scripts/record_clip.sh 60 frames/clip_walk.h264     60 s at 1280x720 @ 30 FPS
#   (same line, no pause: master_service restarts videohub_pc4 within ~2-3 s)
#   W=640 H=360 scripts/record_clip.sh 60 frames/clip_small.h264
# Play back with: build/face_gesture --source frames/clip_walk.h264 ...  (the app cannot run at the same time:
# the camera serves one program)
set -euo pipefail
cd "$(dirname "$0")/.."
SECS=${1:-60}
OUT=${2:-frames/clip_$(date +%H%M%S).h264}
DEV=${DEV:-/dev/video4}
W=${W:-1280}
H=${H:-720}
FPS=${FPS:-30}
BITRATE=${BITRATE:-8000000}
mkdir -p "$(dirname "$OUT")"
echo "recording ${SECS}s of $DEV (${W}x${H} @ ${FPS}) to $OUT (walk around in front of the robot)..."
timeout -s INT "$SECS" gst-launch-1.0 -q -e \
    v4l2src device="$DEV" ! "video/x-raw,format=YUY2,width=$W,height=$H,framerate=$FPS/1" ! \
    nvvidconv ! "video/x-raw(memory:NVMM),format=NV12" ! \
    nvv4l2h264enc bitrate="$BITRATE" insert-sps-pps=true idrinterval="$FPS" ! \
    h264parse ! video/x-h264,stream-format=byte-stream ! filesink location="$OUT" || true
ls -la "$OUT"
