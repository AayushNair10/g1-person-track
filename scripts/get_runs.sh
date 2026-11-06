#!/usr/bin/env bash
# Laptop (Terminal 1): copy every saved run from the robot into results/runs/ (and the offline logs into
# results/logs/), then build the comparison tables and charts. Asks for the robot's password.
#   scripts/get_runs.sh
# Results: results/runs/comparison.md / .csv / .png and results/runs/<run>/analysis.png. results/ is not in git
# (the repository is public and the videos show people).
set -euo pipefail
cd "$(dirname "$0")/.."
HOST=${G1_HOST:-unitree@192.168.123.164}
DIR=${G1_DIR:-aayush_person_track}
mkdir -p results
echo "copying robot:~/$DIR/runs and logs -> results/ (password once per copy)"
scp -q -r "$HOST:$DIR/runs" results/
scp -q -r "$HOST:$DIR/logs" results/ 2>/dev/null || true
PY=${PY:-$HOME/Downloads/benchmark_face/.venv/bin/python}
[ -x "$PY" ] || PY=python3
"$PY" scripts/analyze_runs.py results/runs
echo
echo "open results/runs/comparison.png      (all runs side by side)"
echo "open results/runs/<run>/analysis.png  (one run over time)"
