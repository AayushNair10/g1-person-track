#!/usr/bin/env bash
# Phase 5a (on the robot): offline checks on a recorded clip (scripts/record_clip.sh).
#   scripts/run_offline.sh frames/clip.h264 [sizes...]       sizes default: 640x384 640x640 416x416 320x320
# For each size:
#   1. fp16 vs fp32 on the same input tensors (--compare): detection agreement, box IoU, score difference
#   2. fp16 alone, every frame as fast as possible: the pipeline's maximum frame rate (no camera limit)
# Summaries: logs/compare_<size>.md(.json), logs/offline_<size>.md(.json)
set -uo pipefail
cd "$(dirname "$0")/.."
CLIP=${1:?usage: $0 CLIP.h264 [sizes...]}
shift
SIZES=("$@")
[ ${#SIZES[@]} -eq 0 ] && SIZES=(640x384 640x640 416x416 320x320)
MODEL=${MODEL:-yolo26n}
APP=build/person_track
[ -x "$APP" ] || { echo "build first: make -j4"; exit 1; }

for s in "${SIZES[@]}"; do
    e16=models/${MODEL}_$s.fp16.engine
    e32=models/${MODEL}_$s.fp32.engine
    [ -f "$e16" ] || { echo "missing $e16"; continue; }
    if [ -f "$e32" ]; then
        echo "== $s: fp16 vs fp32"
        "$APP" --engine "$e16" --compare "$e32" --source "$CLIP" --quiet --warmup 5 \
            --label "fp16 vs fp32 $MODEL $s" --summary "logs/compare_${MODEL}_$s.md" 2>&1 | grep -A3 "Agreement" || true
    fi
    echo "== $s: offline throughput (fp16)"
    "$APP" --engine "$e16" --source "$CLIP" --quiet --label "offline $MODEL $s fp16" \
        --csv "logs/offline_${MODEL}_$s.csv" --summary "logs/offline_${MODEL}_$s.md" 2>&1 | grep -E "throughput|inference|latency" || true
done
