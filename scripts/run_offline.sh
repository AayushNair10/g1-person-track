#!/usr/bin/env bash
# Phase 5a (on the robot): offline checks on a recorded clip (scripts/record_clip.sh).
#   scripts/run_offline.sh frames/clip.h264 [sizes...]   sizes default: 640x384 640x640 416x416 416x256 320x320 320x192
# For each size:
#   1. fp16 vs fp32 on the same input tensors (--compare): detection agreement, box IoU, score difference
#   2. fp16 alone, every frame as fast as possible: the pipeline's maximum frame rate (no camera limit)
# Then the compute savings, each on the same clip (only for engines that exist):
#   --hw-resize (VIC scales the frame first) for 640x384 / 416x256 / 320x192, --every 2, lean and DLA engines
# and every run's target compared with the 640x384 fp16 baseline (scripts/compare_runs.py).
# Summaries: logs/compare_*.md(.json), logs/offline_*.md(.json) + .csv, logs/agreement.md(.json)
set -uo pipefail
cd "$(dirname "$0")/.."
CLIP=${1:?usage: $0 CLIP.h264 [sizes...]}
shift
SIZES=("$@")
[ ${#SIZES[@]} -eq 0 ] && SIZES=(640x384 640x640 416x416 416x256 320x320 320x192)
MODEL=${MODEL:-yolo26n}
APP=build/person_track
[ -x "$APP" ] || { echo "build first: make -j4"; exit 1; }

offline() {  # tag engine [app options...]
    local tag=$1 eng=$2
    shift 2
    [ -f "$eng" ] || { echo "   (no $eng, skipped)"; return; }
    echo "== offline $tag"
    "$APP" --engine "$eng" --source "$CLIP" --quiet --label "offline $tag" \
        --csv "logs/offline_$tag.csv" --summary "logs/offline_$tag.md" "$@" 2>&1 | grep -E "throughput|inference|latency" || true
}

for s in "${SIZES[@]}"; do
    e16=models/${MODEL}_$s.fp16.engine
    e32=models/${MODEL}_$s.fp32.engine
    [ -f "$e16" ] || { echo "missing $e16"; continue; }
    if [ -f "$e32" ]; then
        echo "== $s: fp16 vs fp32"
        "$APP" --engine "$e16" --compare "$e32" --source "$CLIP" --quiet --warmup 5 \
            --label "fp16 vs fp32 $MODEL $s" --summary "logs/compare_${MODEL}_$s.md" 2>&1 | grep -A3 "Agreement" || true
    fi
    offline "${MODEL}_$s" "$e16"
done

# compute savings
for s in 640x384 416x256 320x192; do
    offline "${MODEL}_${s}_hw" "models/${MODEL}_$s.fp16.engine" --hw-resize
done
offline "${MODEL}_640x384_every2" "models/${MODEL}_640x384.fp16.engine" --every 2
for e in models/${MODEL}_*.fp16.lean*.engine models/${MODEL}_*.fp16*.dla*.engine; do
    [ -f "$e" ] || continue
    tag=$(basename "$e" .engine)
    [ -n "${done_tags:-}" ] && case " $done_tags " in *" $tag "*) continue ;; esac
    done_tags="${done_tags:-} $tag"
    offline "$tag" "$e"
    offline "${tag}_hw" "$e" --hw-resize
done

base=logs/offline_${MODEL}_640x384.csv
if [ -f "$base" ]; then
    echo "== target agreement with $base"
    python3 scripts/compare_runs.py "$base" logs/offline_*.csv --md logs/agreement.md --json logs/agreement.json
fi
