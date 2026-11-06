#!/usr/bin/env bash
# Laptop -> robot: copies this project into /home/unitree/aayush_person_track/ and nothing else.
#   scripts/deploy.sh code     src/, Makefile, scripts/            (small)
#   scripts/deploy.sh models   models/*.onnx                       (~100 MB)
#   scripts/deploy.sh clip F   one local video file -> frames/     (optional offline test input)
#   scripts/deploy.sh all      code + models
# Same-name files on the robot are overwritten (only inside aayush_person_track/). Prints every step first.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/g1_env.sh
g1_check
what=${1:-all}
DEST="$G1_HOST:$G1_DIR"

echo "Will run on the robot: mkdir -p ~/$G1_DIR/{models,src,build,logs,scripts,frames}"
case "$what" in
    code)   echo "Will copy: src/ Makefile scripts/ -> ~/$G1_DIR/" ;;
    models) echo "Will copy: $(ls models/*.onnx | wc -l | tr -d ' ') ONNX files ($(du -ch models/*.onnx | tail -1 | cut -f1)) -> ~/$G1_DIR/models/" ;;
    clip)   echo "Will copy: ${2:?file} -> ~/$G1_DIR/frames/" ;;
    all)    echo "Will copy: src/ Makefile scripts/ and models/*.onnx -> ~/$G1_DIR/" ;;
    *) echo "usage: $0 code|models|clip FILE|all"; exit 2 ;;
esac
if [ "${YES:-}" != 1 ]; then
    read -r -p "Proceed? [y/N] " ok
    [ "$ok" = y ] || [ "$ok" = Y ] || { echo "cancelled"; exit 1; }
fi

g1_ssh "mkdir -p ~/$G1_DIR/models ~/$G1_DIR/src ~/$G1_DIR/build ~/$G1_DIR/logs ~/$G1_DIR/scripts ~/$G1_DIR/frames"
# no -p: files get the robot's clock (it may read 1970), so make sees consistent times
if [ "$what" = code ] || [ "$what" = all ]; then
    scp -q "${G1_SSH_OPTS[@]}" src/*.cpp src/*.hpp src/*.cu "$DEST/src/"
    scp -q "${G1_SSH_OPTS[@]}" Makefile "$DEST/"
    scp -q "${G1_SSH_OPTS[@]}" scripts/*.sh scripts/make_results.py scripts/compare_runs.py scripts/analyze_runs.py "$DEST/scripts/"
    g1_ssh "chmod +x ~/$G1_DIR/scripts/*.sh"
fi
if [ "$what" = models ] || [ "$what" = all ]; then
    scp -q "${G1_SSH_OPTS[@]}" models/*.onnx "$DEST/models/"
fi
if [ "$what" = clip ]; then
    scp -q "${G1_SSH_OPTS[@]}" "$2" "$DEST/frames/"
fi
echo "done. On the robot:"
g1_ssh "cd ~/$G1_DIR && ls -la . src models frames | head -60"
