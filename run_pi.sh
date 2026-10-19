#!/usr/bin/env bash
# Start the whole thing on the Raspberry Pi: the backend (Whisper + wake word + /app for the phone) and the headset microphone + speaker beeps.
#   cd FALLBACK_CAMOTE && bash run_pi.sh
# Stop with Ctrl+C (both parts stop).
#
# Settings you may need (set before the command, e.g.  CAPTURE_DEVICE=plughw:1,0 bash run_pi.sh):
#   FALLBACK_WHISPER_DIR  folder of the run13 CTranslate2 model   (default: ../03_models/run13-ct2-int8)
#   CAPTURE_DEVICE        microphone, see  python3 pi_capture.py --list   (default: default)
#   BEEP_DEVICE           speaker for the beeps                    (default: default)
#   BEEP_LEVEL            0.1 .. 1.0                               (default: 0.9)
#   PYTHON                the python with the packages installed   (default: python3)
set -u
cd "$(dirname "$0")"
PY="${PYTHON:-python3}"

export FALLBACK_HOST="${FALLBACK_HOST:-0.0.0.0}"        # the phone reaches the Pi on this port (8765)
export FALLBACK_PORT="${FALLBACK_PORT:-8765}"

"$PY" server.py &
SERVER=$!
trap 'kill $SERVER 2>/dev/null; wait 2>/dev/null' EXIT INT TERM

# wait until the backend answers (Whisper takes a few seconds to load)
for _ in $(seq 1 120); do
  if "$PY" -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${FALLBACK_PORT}/status', timeout=1)" >/dev/null 2>&1
  then break; fi
  kill -0 $SERVER 2>/dev/null || { echo "backend stopped, see the message above"; exit 1; }
  sleep 1
done
echo "backend is up on port ${FALLBACK_PORT}; Pi address(es): $(hostname -I 2>/dev/null)"

CAPTURE_URL="ws://127.0.0.1:${FALLBACK_PORT}/ws" "$PY" pi_capture.py
