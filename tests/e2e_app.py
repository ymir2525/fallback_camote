"""The phone-app side of the backend, without a phone. Needs the server running (run.ps1).  Run:  <python> tests\\e2e_app.py     (about 1 minute)

Plays the two roles over real websockets:
  capture page  (/ws)   = the PC page / later the Pi: sends audio, presses nothing
  phone app     (/app)  = the Expo app: listens, presses Start / Scan / End buttons, saves the ticket
Checks: GET /status, hello + state, the app sees wake/case/command/violations events (never the heard text), the app's button commands reach the capture side,
        a new case is refused while a transcript is being made, and the app can save the ticket.
Writes sessions\\<sid> while testing and removes them again.
"""
import asyncio
import json
import os
import shutil
import sys

import httpx
import numpy as np
import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get("FALLBACK_URL", "127.0.0.1:8765")
ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"   {info}" if info else ""), flush=True)


async def drain(ws, store, stop):
    try:
        while not stop.is_set():
            try:
                store.append(json.loads(await asyncio.wait_for(ws.recv(), 0.3)))
            except asyncio.TimeoutError:
                pass
    except websockets.ConnectionClosed:
        pass


async def wait_for(store, pred, secs):
    for _ in range(int(secs / 0.1)):
        hit = next((e for e in store if pred(e)), None)
        if hit:
            return hit
        await asyncio.sleep(0.1)
    return None


async def main():
    async with httpx.AsyncClient(timeout=20) as cl:
        st = (await cl.get(f"http://{BASE}/status")).json()
        check("GET /status answers like the Pi ({state: connected})", st.get("state") == "connected" and st.get("ws") == "/app", str(st)[:120])

    app_ev, cap_ev, stop = [], [], asyncio.Event()
    async with websockets.connect(f"ws://{BASE}/app", max_size=None) as app, websockets.connect(f"ws://{BASE}/ws", max_size=None) as cap:
        t1 = asyncio.create_task(drain(app, app_ev, stop))
        t2 = asyncio.create_task(drain(cap, cap_ev, stop))
        hello = await wait_for(app_ev, lambda e: e["type"] == "hello", 5)
        check("app gets a hello with the state", hello and hello.get("role") == "app" and "state" in hello and hello["commands"][0] == "start_recording")

        # button before anything is listening
        await app.send(json.dumps({"type": "command", "name": "start_recording"}))
        err = await wait_for(app_ev, lambda e: e["type"] == "error", 3)
        check("a button pressed while nothing is listening is answered with an error", err is not None)

        # the capture side starts sending audio (quiet noise): now it is listening
        rng = np.random.default_rng(1)
        silence = (rng.standard_normal(1600) * 8).astype(np.int16).tobytes()
        for _ in range(5):
            await cap.send(silence)
            await asyncio.sleep(0.1)
        lis = await wait_for(app_ev, lambda e: e["type"] == "listening" and e.get("on") is True, 3)
        check("app is told the microphone side is listening", lis is not None)

        # phone presses Start
        await app.send(json.dumps({"type": "command", "name": "start_recording"}))
        case = await wait_for(app_ev, lambda e: e["type"] == "case" and e.get("state") == "started", 5)
        check("phone Start button opens a case (app and capture page both see it)", case is not None and any(e["type"] == "case" for e in cap_ev))
        sid = case["sid"] if case else None
        cmd = next((e for e in app_ev if e["type"] == "command" and e["name"] == "start_recording"), None)
        check("command event has source 'app' and no heard text is sent to the app", cmd and cmd["source"] == "app" and not any(e["type"] in ("line", "segment", "vad") for e in app_ev) and not any("text" in e for e in app_ev if e["type"] == "command"))

        await app.send(json.dumps({"type": "command", "name": "scan_license"}))
        await app.send(json.dumps({"type": "command", "name": "scan_cr"}))
        await app.send(json.dumps({"type": "command", "name": "capture"}))
        got = await wait_for(app_ev, lambda e: e["type"] == "command" and e["name"] == "capture", 3)
        names = [e["name"] for e in app_ev if e["type"] == "command"]
        check("scan_license / scan_cr / capture reach the app, in order", got is not None and names[-3:] == ["scan_license", "scan_cr", "capture"], str(names))

        await app.send(json.dumps({"type": "ping"}))
        pong = await wait_for(app_ev, lambda e: e["type"] == "pong", 3)
        check("pong carries the state (case open)", pong and pong["state"]["case"] and pong["state"]["case"]["sid"] == sid)

        # the capture side now "hears" the sample conversation (violation lines + the spoken 'Kaso tapos na') so the case has real audio to transcribe
        import wave
        with wave.open(os.path.join(os.path.dirname(HERE), "samples", "sample_conversation.wav")) as w:
            pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        for i in range(0, len(pcm), 1600):
            await cap.send(pcm[i:i + 1600].tobytes())
            await asyncio.sleep(0.03)
        for _ in range(40):
            await cap.send(silence)
            await asyncio.sleep(0.05)
        end = await wait_for(app_ev, lambda e: e["type"] == "case" and e.get("state") == "ended", 150)
        check("the case closes (spoken 'Kaso tapos na' in the sample)", end is not None and end["sid"] == sid)
        viol = [e for e in app_ev if e["type"] == "violations"]
        check("the app received the violation list while the case was open", len(viol) >= 1, str(viol[-1]["labels"]) if viol else "")

        # save the ticket from the app -> the transcript starts, announced at once
        await app.send(json.dumps({"type": "save_ticket", "sid": sid, "ticket": {"control_no": "CL-TEST", "violations": []}}))
        saved = await wait_for(app_ev, lambda e: e["type"] == "saved", 6)
        check("app can save the ticket", saved and saved["sid"] == sid, str(saved))
        fs = await wait_for(app_ev, lambda e: e["type"] == "final_start", 5)
        check("transcription start is announced to the app", fs is not None)
        async with httpx.AsyncClient(timeout=20) as cl:
            st = (await cl.get(f"http://{BASE}/status")).json()
        running = sid in st["transcribing"]
        check("status shows the transcript in progress", running, str(st["transcribing"]))
        # a new case while the transcript is running must be refused
        await app.send(json.dumps({"type": "command", "name": "start_recording"}))
        await asyncio.sleep(1.0)
        refused = [e for e in app_ev if e["type"] == "command" and e["name"] == "start_recording" and "transcript" in (e.get("note") or "")]
        check("a new case is refused while the transcript is being made", len(refused) >= 1)

        done = await wait_for(app_ev, lambda e: e["type"] in ("final_done", "final_error"), 90)
        check("transcript finished", done and done["type"] == "final_done", str(done)[:150] if done and done["type"] != "final_done" else "")
        await app.send(json.dumps({"type": "command", "name": "start_recording"}))
        again = await wait_for(app_ev, lambda e: e["type"] == "case" and e.get("state") == "started" and e["sid"] != sid, 5)
        check("after the transcript is done a new case can start", again is not None)
        await app.send(json.dumps({"type": "command", "name": "end_recording"}))
        end2 = await wait_for(app_ev, lambda e: e["type"] == "case" and e.get("state") == "ended" and again and e["sid"] == again["sid"], 6)
        check("phone End button closes the case", end2 is not None)
        stop.set()
        await asyncio.gather(t1, t2)
        for e in app_ev:
            if e["type"] == "case" and e.get("sid"):
                shutil.rmtree(os.path.join(os.path.dirname(HERE), "sessions", e["sid"]), ignore_errors=True)
    print("\nALL PASS" if ok else "\nSOME FAILED")
    sys.exit(0 if ok else 1)


asyncio.run(main())
