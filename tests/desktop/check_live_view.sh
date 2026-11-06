#!/usr/bin/env bash
# Desktop check of the live view's HTTP side: builds tests/desktop/test_live_view.cpp (+ src/live_view.cpp)
# against the OpenCV 4.2 headers (image work stubbed), then fetches /, /snapshot.jpg, /stream and a 404 with curl.
#   OPENCV_MODULES=.../opencv-4.2.0/modules tests/desktop/check_live_view.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
: "${OPENCV_MODULES:?}"
PORT=${PORT:-18090}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"; [ -n "${PID:-}" ] && kill $PID 2>/dev/null; true' EXIT
mkdir -p "$TMP/stub/opencv2" build/desktop
printf '#define HAVE_OPENCV_CORE\n#define HAVE_OPENCV_IMGPROC\n#define HAVE_OPENCV_VIDEOIO\n#define HAVE_OPENCV_IMGCODECS\n' > "$TMP/stub/opencv2/opencv_modules.hpp"
INC=(-isystem "$OPENCV_MODULES/core/include" -isystem "$OPENCV_MODULES/imgproc/include"
     -isystem "$OPENCV_MODULES/videoio/include" -isystem "$OPENCV_MODULES/imgcodecs/include" -isystem "$TMP/stub")
clang++ -std=c++17 -O1 -g -Wall -Wextra -Wno-deprecated-declarations -fsanitize=undefined "${INC[@]}" \
    tests/desktop/test_live_view.cpp src/live_view.cpp -o build/desktop/test_live_view -lpthread

mkfifo "$TMP/in"
build/desktop/test_live_view "$PORT" <"$TMP/in" >"$TMP/out" 2>"$TMP/err" &
PID=$!
exec 3>"$TMP/in"
for _ in $(seq 50); do grep -q "^ready" "$TMP/out" 2>/dev/null && break; sleep 0.1; done
status=0
check() { if eval "$2"; then echo "ok    $1"; else echo "FAIL  $1"; status=1; fi; }
B="http://127.0.0.1:$PORT"
check "port in use: second view switches off, program goes on" 'grep -q "first ok=1 second ok=0" "$TMP/out"'
echo >&3; sleep 0.3
check "no browser open: no frames handed over or encoded" 'tail -1 "$TMP/out" | grep -q "submitted=0 encoded=0 viewers=0"'
check "page /" 'curl -s -m 3 "$B/" | grep -q "<img src=\"/stream\""'
check "404 for other paths" '[ "$(curl -s -m 3 -o /dev/null -w "%{http_code}" "$B/nope")" = 404 ]'
check "snapshot /snapshot.jpg" 'curl -s -m 3 -D "$TMP/h" "$B/snapshot.jpg" | grep -q FAKEJPEG && grep -qi "content-type: image/jpeg" "$TMP/h"'
curl -s -m 2 "$B/stream" >"$TMP/stream" || true
n=$(grep -c -- "--frame" "$TMP/stream" || true)
check "stream /stream: 2 s at 15 FPS gives 20-32 frames (got $n)" '[ "$n" -ge 20 ] && [ "$n" -le 32 ]'
curl -s -m 1 -D "$TMP/h2" -o /dev/null "$B/stream" || true
check "stream headers" 'grep -q "multipart/x-mixed-replace; boundary=frame" "$TMP/h2"'
sleep 0.5; echo >&3; sleep 0.3   # the server notices a closed browser at its next frame (~70 ms)
check "browser closed again: viewers back to 0" 'tail -1 "$TMP/out" | grep -q "viewers=0"'
curl -s -m 10 "$B/stream" >/dev/null &
sleep 0.5
exec 3>&-   # shut the program down while a stream is still open
wait $PID || status=1
PID=
check "shutdown with a browser still connected: quick, clean" 'grep -q "view destroyed in 0\.\|view destroyed in 1\." "$TMP/out"'
check "no undefined-behaviour reports" '! grep -q "runtime error" "$TMP/err"'
grep -h "destroyed\|runtime error" "$TMP/out" "$TMP/err" | head -5
exit $status
