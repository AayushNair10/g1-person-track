#!/usr/bin/env bash
# Phase 3 (on the robot): build TensorRT engines from the ONNX files with trtexec and measure them.
#   cd ~/aayush_person_track && scripts/build_engines.sh                 # default set below
#   scripts/build_engines.sh models/yolo26n-raw_640x384.onnx             # specific files
#   PRECS=fp16 scripts/build_engines.sh ...                              # only fp16
#   FORCE=1 ...                                                          # rebuild existing engines
# Engines go to models/<name>.<fp16|fp32>.engine, logs to logs/trtexec_<name>.<prec>.log, and a table of
# trtexec's GPU compute times (measured with Unitree's services running, CUDA graph like the app) to
# logs/engines.md. Writes nothing outside ~/aayush_person_track.
set -uo pipefail
cd "$(dirname "$0")/.."
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
PRECS=${PRECS:-"fp16 fp32"}
mkdir -p logs models

if [ $# -gt 0 ]; then
    files=("$@")
else
    files=(models/yolo26n_640x384.onnx models/yolo26n_640x640.onnx models/yolo26n_416x416.onnx
           models/yolo26n_416x256.onnx models/yolo26n_320x320.onnx models/yolo26n_320x192.onnx)
fi

for onnx in "${files[@]}"; do
    [ -f "$onnx" ] || { echo "missing $onnx"; continue; }
    name=$(basename "$onnx" .onnx)
    for p in $PRECS; do
        eng=models/$name.$p.engine
        log=logs/trtexec_$name.$p.log
        if [ -s "$eng" ] && [ -z "${FORCE:-}" ]; then
            echo "skip  $eng (exists; FORCE=1 rebuilds)"
            continue
        fi
        flag=""
        [ "$p" = fp16 ] && flag="--fp16"
        echo "build $eng  (log $log)"
        t0=$(date +%s)
        "$TRTEXEC" --onnx="$onnx" $flag --saveEngine="$eng" --timingCacheFile=models/timing.cache \
            --useCudaGraph --warmUp=1000 --duration=10 >"$log" 2>&1
        rc=$?
        echo "      exit $rc after $(( $(date +%s) - t0 )) s"
        if [ $rc -ne 0 ] || [ ! -s "$eng" ]; then
            echo "      FAILED. Last errors:"
            grep -iE "error|fail|unsupported|importFallback" "$log" | tail -8 | sed 's/^/        /'
            rm -f "$eng"
        fi
    done
done

# Table of every engine that has a log
{
    echo "| engine | GPU compute median ms | mean ms | p99 ms | throughput qps | engine MB |"
    echo "|---|---|---|---|---|---|"
    for log in logs/trtexec_*.log; do
        [ -f "$log" ] || continue
        e=${log#logs/trtexec_}; e=${e%.log}
        line=$(grep "GPU Compute Time:" "$log" | tail -1)
        med=$(echo "$line" | grep -o "median = [0-9.]*" | grep -o "[0-9.]*$")
        mean=$(echo "$line" | grep -o "mean = [0-9.]*" | grep -o "[0-9.]*$")
        p99=$(echo "$line" | grep -o "percentile(99%) = [0-9.]*" | grep -o "[0-9.]*$")
        qps=$(grep "Throughput:" "$log" | tail -1 | grep -o "[0-9.]* qps" | grep -o "^[0-9.]*")
        size=""
        [ -f "models/$e.engine" ] && size=$(du -m "models/$e.engine" | cut -f1)
        echo "| $e | ${med:-failed} | ${mean:-} | ${p99:-} | ${qps:-} | ${size} |"
    done
} | tee logs/engines.md
