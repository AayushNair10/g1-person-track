#!/usr/bin/env bash
# Phase 3 (on the robot): build TensorRT engines from the ONNX files with trtexec and measure them.
#   cd ~/aayush_person_track && scripts/build_engines.sh                 # default set below (face + hand models)
#   scripts/build_engines.sh models/yunet_320x192.onnx                   # specific files
#   PRECS="fp16 fp32" scripts/build_engines.sh ...                       # also fp32 (for comparisons)
#   FORCE=1 ...                                                          # rebuild existing engines
#   LEAN=1 PRECS=fp16 scripts/build_engines.sh ...    # no cuDNN/cuBLAS tactics: TensorRT then never loads those
#                                                     # big libraries (less RAM, faster start) -> *.fp16.lean.engine
#   DLA=0 scripts/build_engines.sh ...                # for DLA core 0 (fp16; layers the DLA cannot run fall back
#                                                     # to the GPU) -> *.fp16.dla0.engine; leaves the GPU free
# Engines go to models/<name>.<fp16|fp32>[.lean][.dlaN].engine, logs to logs/trtexec_<engine name>.log, and a table of
# trtexec's GPU compute times (measured with Unitree's services running, CUDA graph like the app) to
# logs/engines.md. Writes nothing outside ~/aayush_person_track.
set -uo pipefail
cd "$(dirname "$0")/.."
TRTEXEC=${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}
PRECS=${PRECS:-fp16}
LEAN=${LEAN:-}
DLA=${DLA:-}
mkdir -p logs models

suffix=""
variant=()
graph=(--useCudaGraph)
if [ -n "$LEAN" ]; then
    suffix+=".lean"
    variant+=(--tacticSources=-CUDNN,-CUBLAS,-CUBLAS_LT)
fi
if [ -n "$DLA" ]; then
    suffix+=".dla$DLA"
    variant+=(--useDLACore="$DLA" --allowGPUFallback)
    graph=()  # DLA work is not recorded into CUDA graphs; time it plainly
fi

if [ $# -gt 0 ]; then
    files=("$@")
else
    files=(models/yunet_640x384.onnx models/palm_192.onnx models/handpose_224.onnx
           models/yunet_416x256.onnx models/yunet_320x192.onnx)
fi

for onnx in "${files[@]}"; do
    [ -f "$onnx" ] || { echo "missing $onnx"; continue; }
    name=$(basename "$onnx" .onnx)
    for p in $PRECS; do
        if [ -n "$DLA" ] && [ "$p" != fp16 ]; then
            echo "skip  $name $p (the DLA runs fp16 / int8 only)"
            continue
        fi
        eng=models/$name.$p$suffix.engine
        log=logs/trtexec_$name.$p$suffix.log
        if [ -s "$eng" ] && [ -z "${FORCE:-}" ]; then
            echo "skip  $eng (exists; FORCE=1 rebuilds)"
            continue
        fi
        flag=""
        [ "$p" = fp16 ] && flag="--fp16"
        echo "build $eng  (log $log)"
        t0=$(date +%s)
        "$TRTEXEC" --onnx="$onnx" $flag ${variant[@]+"${variant[@]}"} --saveEngine="$eng" --timingCacheFile=models/timing.cache \
            ${graph[@]+"${graph[@]}"} --warmUp=1000 --duration=10 >"$log" 2>&1
        rc=$?
        echo "      exit $rc after $(( $(date +%s) - t0 )) s"
        if [ $rc -ne 0 ] || [ ! -s "$eng" ]; then
            echo "      FAILED. Last errors:"
            grep -iE "error|fail|unsupported|importFallback" "$log" | tail -8 | sed 's/^/        /'
            rm -f "$eng"
        elif [ -n "$DLA" ]; then
            echo "      layers on the DLA: $(grep -c '\[DlaLayer\]' "$log"), on the GPU: $(grep -c '\[GpuLayer\]' "$log") (lists in $log)"
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
        line=$(grep "GPU Compute Time: min" "$log" | tail -1)
        med=$(echo "$line" | grep -o "median = [0-9.]*" | grep -o "[0-9.]*$")
        mean=$(echo "$line" | grep -o "mean = [0-9.]*" | grep -o "[0-9.]*$")
        p99=$(echo "$line" | grep -o "percentile(99%) = [0-9.]*" | grep -o "[0-9.]*$")
        qps=$(grep "Throughput:" "$log" | tail -1 | grep -o "[0-9.]* qps" | grep -o "^[0-9.]*")
        size=""
        [ -f "models/$e.engine" ] && size=$(du -m "models/$e.engine" | cut -f1)
        echo "| $e | ${med:-failed} | ${mean:-} | ${p99:-} | ${qps:-} | ${size} |"
    done
} | tee logs/engines.md
