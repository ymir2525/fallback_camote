# FALLBACK_CAMOTE

A separate, testable copy of the "fallback" idea: **commands and violations stay live; the full transcript is made after the stop, from the recorded video.**
It looks like the mobile ticket screen from `FIGMAsamplefull.png` and runs in the browser.

* Not connected to the CAMOTE fine-tuning / data gathering and not part of `hey-trafficam`. It only **reads** the Whisper model folder
  (`..\03_models\run13-ct2-int8`). The matcher package is a **copy** in `camote_speech\` (same files as the handoff), so nothing existing was changed.
* Nothing is downloaded or installed. It uses the Python in `hey-trafficam\venv` (set `FALLBACK_PYTHON` to use another one).

## Run

```
powershell -File run.ps1          # or:  <python> server.py
```
Wait for `[fallback] ready`, then open **http://127.0.0.1:8765/** in Chrome (localhost allows camera and microphone). Stop it with Ctrl+C.
Phone on the same Wi-Fi: browsers only allow the microphone on `https` or `localhost`, so test on the PC first.

## Try it (no microphone needed)
1. Tap the gear (top right) → **Play sample conversation**. It plays an enforcer + driver stop made from your own Set 3 clips.
2. Or tap **GO LIVE**, allow camera + microphone, and say: *"Hey Jarvis, mag-record ka na."* → violation lines → *"Scan license"*, *"Capture"* → *"Kaso tapos na."*
3. The screen jumps to the step each command belongs to. When the case closes, **Preview** shows the summary and the video. Tap **SAVE TICKET**: the transcript and the cross-check are made then.

## What is live and what is after the stop
| | When | How |
|---|---|---|
| Wake word | live | audio model (openWakeWord "hey_jarvis", threshold 0.3) **or** the word "Jarvis" found in the text of an idle line (loose sound match) |
| 5 commands: Record, Scan license, Scan CR, Capture, End case | live | CAMOTE command matcher on the enforcer's line; buttons on the screen do the same (source "button") |
| Violation tags | live, while a case is open | CAMOTE phonetic fix + violation matcher; denial rule; two neighbouring lines are also read together |
| Live transcript on screen | **removed** | only commands, tags and timing are shown (turn on "Show what was heard" in the tools to see text while testing) |
| Full transcript with times | after you tap SAVE TICKET | the video's audio (read with PyAV, no ffmpeg), Whisper beam 5, same matchers |
| Cross-check | after the stop | live list vs the recording: confirmed / heard live but not found / found only after the stop |
| Speaker labels (enforcer / driver) | **later** | hook is `speaker_of()` in `engine.py`; every line is "-" for now |

Live path details (all in `engine.py` and `vadcut.py`): Silero VAD cutter (150 Hz high-pass on the detector copy only, on ≥ 0.35, line ends after 0.7 s of quiet or 0.45 s right after the wake word,
12 s / 6 s cap), a queue where commands go first, short neighbouring lines sent to Whisper as one call, a repetition filter for Whisper loops, and a "late" flag after 8 s in the queue.

## Files
```
server.py        web server + websocket + upload of the video
engine.py        live path + after-stop pass
vadcut.py        speech cutter (Silero VAD)
static\          the page (index.html, style.css, app.js)
camote_speech\   COPY of the CAMOTE matcher package (unchanged)
samples\         sample_conversation.wav  (your own voice: keep it out of git)
sessions\        every case: case.json, stream_case.wav, video.webm, final_*.json   (private: keep out of git)
tests\           test_cutter.py, e2e_live.py, e2e_ws.py (needs the server running), make_video.py, make_sample.py
```

## How to compare with hey-trafficam
Play the same recording through both and compare: (1) did the same commands and violations fire, (2) the "after you stopped talking" seconds shown by the app
(wait + queue + Whisper), (3) how the screen behaves when a line is long or noisy. **Export report (JSON)** in the Preview saves all numbers.
Settings to try (gear): wake mode (`audio + text`, `audio only`, `off`), reading two lines together, speech sensitivity (0.25 for far voices).

## Tests
```
<python> tests\test_cutter.py      # pure road noise = 0 segments; streaming = whole file; lines covered
<python> tests\e2e_live.py         # real clips + road noise in real time: wake, 4 commands, violations, after-stop pass   (E2E_NOISE_GAIN=0.01 for a quiet street)
<python> tests\e2e_ws.py           # same through the real server + a WebM (VP8+Opus) video upload like Chrome's
```

## Known limits (read before trusting numbers)
* **Wake word is the weakest part** on the owner's voice (measured, see the summary in `00_MAP.md`): the audio model alone catches only about 2 of 8 "Jarvis" clips even clean and almost none in road noise; the text check catches 8 of 8 clean but roughly half at light road noise. Use the buttons, wake mode `off`, or a physical button as backup.
* A missed **End case** line leaves the case open until the button is tapped.
* Road noise hurts Whisper (about 10 dB: roughly half the words wrong). This app does not fix that; the noise-trained model (run14) does.
* The camera/microphone path was tested with a synthetic camera + microphone stream, not with a physical device. Real-device test is up to the owner.
* "Scan license / Scan CR / Capture" save a camera frame only. Reading the license text (OCR) is not part of this fallback.
* Speaker labels: not yet.

