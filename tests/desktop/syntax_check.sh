#!/usr/bin/env bash
# Compile-check the robot sources on a machine without CUDA/TensorRT (e.g. the Mac), against the real
# TensorRT 8.5, CUDA 11.4 and OpenCV 4.2 headers. Nothing is linked or run. Needs clang (CUDA parsing).
#   TRT_INC=.../TensorRT(release/8.5)/include  CUDA_INC=.../cuda-11.4/include \
#   OPENCV_MODULES=.../opencv-4.2.0/modules  tests/desktop/syntax_check.sh
# Header sources (download only, nothing installed): git sparse checkouts of NVIDIA/TensorRT release/8.5
# include/ and opencv 4.2.0 modules/{core,imgproc,videoio,imgcodecs}/include; the include/ folders of the NVIDIA
# redist archives cuda_cudart 11.4.148 and cuda_nvcc 11.4.152 merged into one folder.
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${TRT_INC:?}" "${CUDA_INC:?}" "${OPENCV_MODULES:?}"
CXX=${CXX:-clang++}

# generated / unused headers the real installs would provide
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/stub/opencv2" "$TMP/cuda/bin" "$TMP/cuda/lib64" "$TMP/cuda/nvvm/libdevice"
printf '#define HAVE_OPENCV_CORE\n#define HAVE_OPENCV_IMGPROC\n#define HAVE_OPENCV_VIDEOIO\n#define HAVE_OPENCV_IMGCODECS\n' > "$TMP/stub/opencv2/opencv_modules.hpp"
touch "$TMP/stub/curand_mtgp32_kernel.h" "$TMP/cuda/nvvm/libdevice/libdevice.10.bc"
printf '#!/bin/sh\n' > "$TMP/cuda/bin/nvcc"; chmod +x "$TMP/cuda/bin/nvcc"
ln -s "$CUDA_INC" "$TMP/cuda/include"

INC=(-isystem "$TRT_INC" -isystem "$CUDA_INC" -isystem "$OPENCV_MODULES/core/include"
     -isystem "$OPENCV_MODULES/imgproc/include" -isystem "$OPENCV_MODULES/videoio/include"
     -isystem "$OPENCV_MODULES/imgcodecs/include" -isystem "$TMP/stub")
WARN=(-Wall -Wextra -Wno-deprecated-declarations -Wshadow)
status=0
for f in src/*.cpp; do
    if "$CXX" -std=c++17 -fsyntax-only "${WARN[@]}" "${INC[@]}" "$f"; then echo "ok    $f"; else echo "FAIL  $f"; status=1; fi
done
for side in host device; do
    if "$CXX" -x cuda --cuda-path="$TMP/cuda" --cuda-$side-only -nocudalib --cuda-gpu-arch=sm_87 -std=c++17 \
        -fsyntax-only "${WARN[@]}" -isystem "$TMP/stub" -include tests/desktop/cuda_launch_shim.hpp src/preprocess.cu; then
        echo "ok    src/preprocess.cu ($side side)"
    else
        echo "FAIL  src/preprocess.cu ($side side)"; status=1
    fi
done
exit $status
