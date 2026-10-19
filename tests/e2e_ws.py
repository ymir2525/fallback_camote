"""Through the REAL server: your Set 3 clips + road noise are streamed over the WebSocket in real time (like the browser does),
then a test video (WebM, VP8 + Opus, like Chrome's) is uploaded and the after-stop pass reads the video's audio.
Needs the server running (run.ps1). Run:  <python> tests\\e2e_ws.py     (about 3 minutes)
"""
import asyncio
import csv
import glob
import json
import os
import shutil
import sys
import tempfile
import time
import wave

import httpx
import numpy as np
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from make_video import make_webm  # noqa: E402

CAMOTE = os.path.abspath(os.path.join(HERE, "..", ".."))
S01 = os.path.join(CAMOTE, "01_inbox", "owner_sessions", "S01_20261018")
NOISE = os.path.join(CAMOTE, "Bgdataset-1.wav")
SESSIONS = os.path.join(os.path.dirname(HERE), "sessions")
SR = 16000
GAIN = float(os.environ.get("E2E_NOISE_GAIN", "0.03"))
BASE = os.environ.get("FALLBACK_URL", "127.0.0.1:8765")


def read_wav(p):
    with wave.open(p) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]
    if rate != SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(rate, SR)
        x = resample_poly(x.astype(np.float32), SR // g, rate // g).astype(np.int16)
    return x


rows = {r["line_id"]: r for r in csv.DictReader(open(os.path.join(S01, "labels.csv"), encoding="utf-8"))}


def clip(lid):
    return read_wav(sorted(glob.glob(os.path.join(S01, lid + "_*.wav")))[-1])


noise = read_wav(NOISE)
rng = np.random.default_rng(5)


def road(n):
    o = int(rng.integers(0, len(noise) - n - 1))
    return (noise[o:o + n].astype(np.float32) * GAIN).astype(np.int16)


LINES = ["R1005", "R2005", "R2010", "R1017", "R1033", "R1043"]     # "Hey Jarvis, mag-record ka." / Influence / Overspeeding / Scan license / Capture / "Kaso tapos na, Jarvis."
parts = [road(SR * 4)]
for lid in LINES:
    x = clip(lid)
    parts += [np.clip(x.astype(np.int32) + road(len(x)), -32768, 32767).astype(np.int16), road(SR * 3)]
parts.append(road(SR * 3))
stream = np.concatenate(parts)
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"   {info}" if info else ""), flush=True)


async def main():
    events = []
    async with websockets.connect(f"ws://{BASE}/ws", max_size=None) as ws:
        async def reader():
            async for m in ws:
                events.append(json.loads(m))
        rt = asyncio.create_task(reader())
        await asyncio.sleep(0.5)
        await ws.send(json.dumps({"type": "marker", "vid": "v1"}))
        t0 = time.time()
        step = SR // 10
        for i in range(0, len(stream), step):
            await ws.send(stream[i:i + step].tobytes())
            await asyncio.sleep(max(0.0, t0 + (i + step) / SR - time.time()))
        for _ in range(60):
            if any(e["type"] == "case" and e["state"] == "ended" for e in events):
                break
            await asyncio.sleep(0.5)
        end = next((e for e in events if e["type"] == "case" and e["state"] == "ended"), None)
        for e in events:
            if e["type"] == "line":
                print(f"   line @{e['clock']:5.1f}s  {(e.get('text') or '(dropped: ' + str(e.get('dropped')) + ') ' + str(e.get('heard', '')))[:70]}  cmds={e.get('commands')} viol={e.get('violations')}  total {e['timing']['total_s']}s")
            elif e["type"] in ("wake", "error"):
                print("  ", e)
        check("server announced the hello (config, wake word)", events and events[0]["type"] == "hello" and events[0]["wake_word"])
        cmds = [e["name"] for e in events if e["type"] == "command"]
        check("commands came back over the websocket in order", cmds == ["start_recording", "scan_license", "capture", "end_recording"], str(cmds))
        check("case ended with a session id", end is not None and bool(end.get("sid")))
        if end is None:
            rt.cancel()
            return None
        sid = end["sid"]
        print("live violations:", end["violations"], "| session", sid)

        # --- after the stop: upload the "video" like the browser does, wait for the transcript
        tmp = tempfile.mkdtemp()
        vp = os.path.join(tmp, "video.webm")
        make_webm(stream, vp)
        size = os.path.getsize(vp)
        print(f"test video: {size / 1024:.0f} KB webm (vp8+opus)")
        async with httpx.AsyncClient(timeout=600) as cl:
            r = await cl.post(f"http://{BASE}/api/session/{sid}/video", files={"video": ("video.webm", open(vp, "rb"), "video/webm")}, data={"vid": "v1"})
            check("video upload accepted", r.status_code == 200, r.text[:80])
            await asyncio.sleep(2)
            check("no transcript before the ticket is saved", not any(e["type"] == "final_start" for e in events))
            r = await cl.post(f"http://{BASE}/api/session/{sid}/save", json={"control_no": "CL-TEST", "violations": []})
            check("ticket saved, transcript started", r.status_code == 200 and r.json().get("source") == "video", r.text[:80])
            done = None
            for _ in range(240):
                done = next((e for e in events if e["type"] in ("final_done", "final_error") and e["source"] == "video"), None)
                if done:
                    break
                await asyncio.sleep(1)
            check("after-stop pass finished on the video", done is not None and done["type"] == "final_done", str(done)[:200] if done and done["type"] != "final_done" else "")
            if done and done["type"] == "final_done":
                rep = done["report"]
                print(f"video pass: {rep['audio_s']} s audio in {rep['seconds']} s (x{rep['rtf']}), {len(rep['lines'])} lines, beam {rep['beam']}")
                for l in rep["lines"]:
                    print(f"   [{l['t0']:5.1f}] {'*' if l['in_case'] else ' '} {l['text'][:78]}")
                check("video transcript: cross-check has nothing missing", not rep["live_only"], f"live {rep['live_violations']} final {rep['violations']}")
                check("lines carry video-timeline times", all(l["t1"] <= rep["audio_s"] + 1 for l in rep["lines"]))
            meta = (await cl.get(f"http://{BASE}/api/session/{sid}")).json()
            check("session API returns video + final", meta.get("video") and "video" in meta.get("finals", {}))
            rr = await cl.get(f"http://{BASE}/api/session/{sid}/video", headers={"Range": "bytes=0-99"})
            check("video can be played / seeked (Range request works)", rr.status_code in (200, 206) and len(rr.content) > 0, f"status {rr.status_code}")
            ls = (await cl.get(f"http://{BASE}/api/sessions")).json()
            check("past cases list shows it", any(s["sid"] == sid for s in ls))
        rt.cancel()
        shutil.rmtree(tmp, ignore_errors=True)
        return sid


sid = asyncio.run(main())
if sid:
    shutil.rmtree(os.path.join(SESSIONS, sid), ignore_errors=True)
print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
