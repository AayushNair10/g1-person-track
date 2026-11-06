#!/usr/bin/env bash
# On the robot: record the head-camera stream (exactly what videohub multicasts, no re-encoding) for
# offline tests, e.g. fp16 vs fp32 on identical frames. Only listens to the multicast stream, like the app.
#   scripts/record_clip.sh 60 frames/clip_walk.h264
# Play back with: build/person_track --source frames/clip_walk.h264 ...
set -euo pipefail
cd "$(dirname "$0")/.."
SECS=${1:-60}
OUT=${2:-frames/clip_$(date +%H%M%S).h264}
mkdir -p "$(dirname "$OUT")"
echo "recording ${SECS}s of the head-camera stream to $OUT (walk around in front of the robot)..."
timeout -s INT "$SECS" gst-launch-1.0 -q -e \
    udpsrc multicast-group=230.1.1.1 port=1720 multicast-iface=eth0 auto-multicast=true \
    caps="application/x-rtp,media=video,clock-rate=90000,encoding-name=H264,payload=96" ! \
    rtph264depay ! h264parse config-interval=-1 ! video/x-h264,stream-format=byte-stream ! \
    filesink location="$OUT" || true
ls -la "$OUT"
