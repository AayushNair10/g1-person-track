#!/usr/bin/env bash
# Phase 1: read-only checks of the G1's Jetson. Changes nothing on the robot.
# Run from the laptop (the robot needs no copy of this file):
#   ssh -n unitree@192.168.123.164 "$(cat scripts/phase1_verify.sh)"
# sudo is used only non-interactively (sudo -n): if it needs a password the line says so
# and the check is done without sudo where possible.

sec() { printf '\n===== %s =====\n' "$1"; }

sec "model";        cat /proc/device-tree/model; echo
sec "L4T release";  cat /etc/nv_tegra_release
sec "packages";     dpkg -l | grep -E "tensorrt |libnvinfer-dev|libopencv-dev" | awk '{print $2, $3}'
sec "power mode"
sudo -n nvpmodel -q 2>/dev/null || { echo "(sudo -n refused: needs password; trying without sudo)"; nvpmodel -q 2>&1; }
sec "memory / disk"; free -h; df -h /
sec "video devices"
for v in /sys/class/video4linux/video*; do echo "$(basename "$v"): $(cat "$v/name")"; done
sec "video device users"
sudo -n fuser -v /dev/video* 2>&1 || echo "(sudo -n refused or no users; non-root view below)"
ps -eo user,pid,args | grep -E "videohub|master_service" | grep -v grep
sec "TensorRT python"; python3 -c "import tensorrt; print(tensorrt.__version__)"
sec "OpenCV";       pkg-config --modversion opencv4 || pkg-config --modversion opencv
sec "nvv4l2decoder"; gst-inspect-1.0 nvv4l2decoder | head -3

# Extra read-only facts the later phases depend on
sec "extra: tools"
ls -l /usr/src/tensorrt/bin/trtexec /usr/local/cuda/bin/nvcc 2>&1
/usr/local/cuda/bin/nvcc --version | tail -2
g++ --version | head -1
sec "extra: OpenCV build (GStreamer / CUDA)"
python3 -c "import cv2; info=cv2.getBuildInformation(); [print(l.strip()) for l in info.splitlines() if any(k in l for k in ('GStreamer','CUDA','Video I/O','FFMPEG','V4L'))]" 2>/dev/null || opencv_version -v 2>/dev/null | grep -E "GStreamer|CUDA|FFMPEG|V4L" || echo "(no python cv2 / opencv_version to show build info)"
sec "extra: network / multicast"
ip -4 addr show eth0 | grep inet
ip maddr show dev eth0 2>/dev/null | grep -E "230\.1\.1\.1|inet" | head -5
sec "extra: clock / workspace"
date
ls -ld /home/unitree/aayush_person_track 2>&1
sec "extra: GPU load right now"
timeout 3 tegrastats --interval 1000 2>&1 | head -2
