"""Headless capture side for the Pi: the microphone (headset) goes to the backend, the beeps come out of the Pi's speaker.

It does what the browser page does for audio, nothing more: it sends 16 kHz mono audio to ws://127.0.0.1:8765/ws and plays a short beep
when the backend says "wake" / "case started" / "nothing heard after Hey Jarvis". No video yet. The cases are still tagged and transcribed
by the server, and the phone app talks to the server (/app), not to this program.

  python3 pi_capture.py                       microphone via arecord (Raspberry Pi OS has it: package alsa-utils)
  python3 pi_capture.py --list                show the sound cards
  python3 pi_capture.py --wav some.wav        play a WAV file into the backend instead of the microphone (test without a Pi)

Settings (environment variables):
  CAPTURE_URL      ws://127.0.0.1:8765/ws
  CAPTURE_DEVICE   arecord device, "default" (find yours with  arecord -l  ->  plughw:1,0  for card 1)
  BEEP_DEVICE      aplay device for the beeps, "default" (same naming, see aplay -l)
  BEEP_LEVEL       0.1 .. 1.0, default 0.9 (the PC page used 0.25 and it was too quiet)
  BEEPS            1 / 0, default 1
"""
import argparse
import asyncio
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave

import websockets

SR = 16000
CHUNK = 1600                                                     # samples per message (0.1 s), same as the browser page
URL = os.environ.get("CAPTURE_URL", "ws://127.0.0.1:8765/ws")
MIC = os.environ.get("CAPTURE_DEVICE", "default")
SPEAKER = os.environ.get("BEEP_DEVICE", "default")
LEVEL = max(0.05, min(1.0, float(os.environ.get("BEEP_LEVEL", "0.9"))))
BEEPS_ON = os.environ.get("BEEPS", "1") == "1"
BEEP_SR = 22050
BEEP_TONES = {"wake": [(880, 0.12)], "case": [(660, 0.12), (990, 0.16)], "timeout": [(220, 0.30)]}     # same sounds as the browser page


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------- beeps
def make_beep(kind, folder):
    """Write beep_<kind>.wav; returns (path, length in ms)."""
    frames, total = [], 0.0
    for hz, dur in BEEP_TONES[kind]:
        n = int(BEEP_SR * dur)
        for i in range(n):
            fade = min(1.0, i / (0.01 * BEEP_SR), (n - i) / (0.02 * BEEP_SR))                 # short fade in/out: no click
            frames.append(int(32767 * LEVEL * fade * math.sin(2 * math.pi * hz * i / BEEP_SR)))
        frames += [0] * int(BEEP_SR * 0.04)
        total += dur + 0.04
    path = os.path.join(folder, f"beep_{kind}.wav")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(BEEP_SR)
        w.writeframes(struct.pack("<%dh" % len(frames), *frames))
    return path, int(total * 1000)


class Speaker:
    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix="camote_beeps_")
        self.files = {k: make_beep(k, self.dir) for k in BEEP_TONES}
        self.aplay = shutil.which("aplay")
        self.winsound = None
        if not self.aplay and sys.platform == "win32":
            import winsound
            self.winsound = winsound
        if BEEPS_ON and not (self.aplay or self.winsound):
            log("no sound player found (aplay): beeps will only be printed. Install alsa-utils.")

    def play(self, kind):
        """Start the beep (does not wait for it); returns its length in ms."""
        path, ms = self.files[kind]
        if not BEEPS_ON:
            return 0
        log(f"beep: {kind}")
        try:
            if self.aplay:
                cmd = [self.aplay, "-q"] + (["-D", SPEAKER] if SPEAKER != "default" else []) + [path]
                subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            elif self.winsound:
                self.winsound.PlaySound(path, self.winsound.SND_FILENAME | self.winsound.SND_ASYNC)
        except OSError as ex:
            log(f"beep failed: {ex}")
        return ms


# ----------------------------------------------------------------------------------------------------------- microphone
def list_cards():
    for tool, what in (("arecord", "microphones"), ("aplay", "speakers")):
        if shutil.which(tool):
            print(f"--- {what} ({tool} -l)")
            subprocess.run([tool, "-l"])
        else:
            print(f"--- {tool} not found (sudo apt install alsa-utils)")


def mic_reader(loop, queue, stop, wav_path=None):
    """Thread: puts CHUNK-sample int16 blocks (bytes) on the queue."""
    size = CHUNK * 2
    if wav_path:
        with wave.open(wav_path) as w:
            assert w.getframerate() == SR and w.getnchannels() == 1, "the test WAV must be 16 kHz mono"
            data = w.readframes(w.getnframes())
        silence = bytes(size)
        data += silence * 60
        t0 = time.time()
        for i in range(0, len(data) - size + 1, size):
            if stop.is_set():
                return
            wait = t0 + (i // size) * 0.1 - time.time()                                       # real time, like a microphone
            if wait > 0:
                time.sleep(wait)
            loop.call_soon_threadsafe(queue.put_nowait, data[i:i + size])
        loop.call_soon_threadsafe(queue.put_nowait, None)
        return
    if not shutil.which("arecord"):
        log("arecord not found: sudo apt install alsa-utils")
        loop.call_soon_threadsafe(queue.put_nowait, None)
        return
    cmd = ["arecord", "-q", "-D", MIC, "-f", "S16_LE", "-r", str(SR), "-c", "1", "-t", "raw"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        while not stop.is_set():
            buf = proc.stdout.read(size)
            if len(buf) < size:
                log("microphone stopped:", (proc.stderr.read() or b"").decode(errors="replace").strip()[:200])
                break
            loop.call_soon_threadsafe(queue.put_nowait, buf)
    finally:
        proc.kill()
        loop.call_soon_threadsafe(queue.put_nowait, None)


# ------------------------------------------------------------------------------------------------------------- session
class Beeper:
    """Same rules as the browser page: the wake beep waits until you stop talking (so it never lands inside "Hey Jarvis, mag record ka na")."""

    def __init__(self, speaker, send):
        self.speaker, self.send = speaker, send
        self.hearing = False
        self.pending = False
        self.timer = None

    async def beep(self, kind):
        ms = self.speaker.play(kind)
        if ms:
            await self.send({"type": "beep", "ms": ms})                                     # the server ignores that short sound if it leaks into the microphone

    async def wake(self):
        self.pending = True
        if self.timer:
            self.timer.cancel()
        if not self.hearing:
            await self.flush()
        else:
            self.timer = asyncio.get_running_loop().call_later(4.0, lambda: asyncio.ensure_future(self.flush()))

    async def flush(self):
        if self.timer:
            self.timer.cancel()
        if self.pending:
            self.pending = False
            await self.beep("wake")

    def cancel_wake(self):
        self.pending = False
        if self.timer:
            self.timer.cancel()


async def session(args, speaker):
    loop = asyncio.get_running_loop()
    async with websockets.connect(URL, max_size=None, ping_interval=20) as ws:
        log("connected to", URL)

        async def send(obj):
            await ws.send(json.dumps(obj))

        beeper = Beeper(speaker, send)
        queue, stop = asyncio.Queue(maxsize=200), threading.Event()
        threading.Thread(target=mic_reader, args=(loop, queue, stop, args.wav), daemon=True).start()

        async def pump():
            while True:
                buf = await queue.get()
                if buf is None:
                    return
                await ws.send(buf)

        async def events():
            async for raw in ws:
                ev = json.loads(raw)
                t = ev.get("type")
                if t == "hello":
                    log(f"backend ready: whisper={ev.get('whisper')} wake_word={'ok' if ev.get('wake_word') else 'NOT AVAILABLE'}")
                elif t == "vad":
                    beeper.hearing = bool(ev.get("speech"))
                    if not beeper.hearing:
                        await beeper.flush()
                elif t == "wake":
                    log("wake word heard")
                    await beeper.wake()
                elif t == "awake_end":
                    log("nothing heard after the wake word: back to idle")
                    beeper.cancel_wake()
                    await beeper.beep("timeout")
                elif t == "case":
                    log(f"case {ev.get('state')}: {ev.get('sid')}")
                    if ev.get("state") == "started":
                        beeper.cancel_wake()
                        await beeper.beep("case")
                elif t == "command":
                    log(f"command: {ev.get('name')}" + (f" ({ev['note']})" if ev.get("note") else ""))
                elif t == "violations":
                    log("violations:", ", ".join(ev.get("labels", [])) or "-")
                elif t in ("final_start", "final_done", "final_error"):
                    log(t, ev.get("sid", ""), ev.get("message", ""))

        pumping, listening = asyncio.create_task(pump()), asyncio.create_task(events())
        try:
            done, _ = await asyncio.wait([pumping, listening], return_when=asyncio.FIRST_COMPLETED)
            if pumping in done and args.wav and not listening.done():
                await asyncio.sleep(args.after)                                             # test file finished: keep listening for the last events
            for d in done:
                d.result()
        finally:
            stop.set()
            pumping.cancel()
            listening.cancel()
            try:
                await ws.send(json.dumps({"type": "listening", "on": False}))
            except Exception:
                pass


async def main(args):
    speaker = Speaker()
    if args.beep_test:
        for k in BEEP_TONES:
            speaker.play(k)
            await asyncio.sleep(1.2)
        return
    while True:
        try:
            await session(args, speaker)
            if args.wav:
                return
        except (OSError, websockets.WebSocketException) as ex:
            log(f"backend not reachable ({ex.__class__.__name__}); retrying in 2 s")
        await asyncio.sleep(2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="show the sound cards and stop")
    ap.add_argument("--beep-test", action="store_true", help="play the three beeps on the speaker and stop")
    ap.add_argument("--wav", help="16 kHz mono WAV to play into the backend instead of the microphone")
    ap.add_argument("--after", type=float, default=15, help="with --wav: seconds to keep listening to events after the file ended")
    a = ap.parse_args()
    if a.list:
        list_cards()
        sys.exit(0)
    try:
        asyncio.run(main(a))
    except KeyboardInterrupt:
        pass
