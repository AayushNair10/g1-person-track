#!/usr/bin/env bash
# Build build/face_gesture on the robot without make: the same commands as the Makefile, with the same
# JetPack 5.1.1 tools FaceHandTrack's build.sh uses (nvcc, g++, OpenCV 4.2 via pkg-config, TensorRT, CUDA runtime).
# Nothing is installed or downloaded; it only writes build/.
#   scripts/build.sh          ->  build/face_gesture
set -euo pipefail
cd "$(dirname "$0")/.."
CUDA=${CUDA:-/usr/local/cuda}
mkdir -p build
echo "[1/2] CUDA kernel (nvcc)..."
"$CUDA/bin/nvcc" -std=c++17 -O3 -arch=sm_87 -Xcompiler -Wall -c src/preprocess.cu -o build/preprocess.o
echo "[2/2] app (g++, about a minute)..."
g++ -std=c++17 -O3 -Wall -Wextra -Wno-deprecated-declarations -Wno-format-truncation \
    -I"$CUDA/include" $(pkg-config --cflags opencv4) \
    src/main.cpp src/trt_engine.cpp src/camera.cpp src/ocsort.cpp src/video_sink.cpp src/live_view.cpp src/gestures.cpp \
    build/preprocess.o \
    -o build/face_gesture \
    -L"$CUDA/lib64" -lnvinfer -lcudart $(pkg-config --libs opencv4) -lpthread
echo "built build/face_gesture"
