#!/usr/bin/env bash
# Robot -> laptop: copies ~/aayush_person_track/logs to results/robot_<time>/, builds RESULTS.md there
# (with the Phase 1 output from results/phase1.txt if present), copies it to docs/RESULTS.md for git, and
# puts RESULTS.md back into the robot's logs/ folder too (the only write on the robot).
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/g1_env.sh
g1_check
dst=results/robot_$(date +%Y%m%d_%H%M%S)
mkdir -p "$dst"
echo "copying robot:~/$G1_DIR/logs -> $dst/"
scp -q -r "${G1_SSH_OPTS[@]}" "$G1_HOST:$G1_DIR/logs" "$dst/"
[ -f results/phase1.txt ] && cp results/phase1.txt "$dst/logs/phase1.txt"
python3 scripts/make_results.py "$dst/logs"
mkdir -p docs
cp "$dst/logs/RESULTS.md" docs/RESULTS.md
echo "copying RESULTS.md -> robot:~/$G1_DIR/logs/RESULTS.md"
scp -q "${G1_SSH_OPTS[@]}" "$dst/logs/RESULTS.md" "$G1_HOST:$G1_DIR/logs/RESULTS.md"
echo "done: $dst/logs/RESULTS.md (also docs/RESULTS.md)"
