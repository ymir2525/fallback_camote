"""Wake-word-only mode (Whisper idle until "Hey Jarvis"), measured with your real recordings, limited to 4 cores like a Raspberry Pi 5.

  A  one breath  "Hey Jarvis, mag-record ka"            -> wake heard? start_recording heard? how fast?
  B  with a pause "Hey Jarvis." ... 1 s ... "mag-record"  -> same
  C  idle talk (no wake word, road noise)                -> Whisper must not run; false wakes; CPU used while idle
Each is repeated clean and with road noise. Run:  <python> tests\\e2e_wake.py        (about 15 minutes, real time)
Env: E2E_NOISE="0,0.01,0.05" noise gains, E2E_MAX_CLIPS=12, FALLBACK_CORES=4 (default here).
Nothing is written to the project (the case it opens is removed again).
"""
import csv
import glob
import os
import shutil
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ.setdefault("FALLBACK_CORES", "4")
os.environ.setdefault("FALLBACK_LOG", "0")
import engine as E  # noqa: E402
import limits  # noqa: E402
from camote_speech import analyze  # noqa: E402

CAMOTE = os.path.abspath(os.path.join(HERE, "..", ".."))
OWN = os.path.join(CAMOTE, "01_inbox", "owner_sessions")
NOISE = os.path.join(CAMOTE, "Bgdataset-1.wav")
SR = 16000
GAINS = [float(x) for x in os.environ.get("E2E_NOISE", "0,0.01,0.05").split(",")]
MAXC = int(os.environ.get("E2E_MAX_CLIPS", "12"))


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


def load_rows(session):
    d = os.path.join(OWN, session)
    out = {}
    for r in csv.DictReader(open(os.path.join(d, "labels.csv"), encoding="utf-8")):
        f = sorted(glob.glob(os.path.join(d, r["line_id"] + "_*.wav")))
        if f:
            out[r["line_id"]] = (f[-1], r["text"], r.get("role", ""))
    return out


rows = {}
for s in ("S01_20261018", "S01_20261017", "S01_priority"):
    for k, v in load_rows(s).items():
        rows.setdefault(s + ":" + k, v)

one_breath, jarvis_alone, command_only, idle = [], [], [], []
for k, (f, text, role) in rows.items():
    low = text.lower()
    a = analyze(text)
    if "jarvis" in low and a["commands"][:1] == ["start_recording"]:
        one_breath.append((k, f, text))
    elif low.strip(" .,!") in ("hey jarvis", "jarvis", "okay jarvis"):
        jarvis_alone.append((k, f, text))
    elif "jarvis" not in low and a["commands"][:1] == ["start_recording"] and role == "OFFICER":
        command_only.append((k, f, text))
    elif "jarvis" not in low and not a["commands"] and role in ("OFFICER", "DRIVER") and 1.5 < len(text) / 14 < 9:
        idle.append((k, f, text))
one_breath, jarvis_alone, command_only = one_breath[:MAXC], jarvis_alone[:6], command_only[:6]
idle = idle[:25]
print(f"clips: one-breath {len(one_breath)}, 'Hey Jarvis' alone {len(jarvis_alone)}, command only {len(command_only)}, idle talk {len(idle)}", flush=True)

noise = read_wav(NOISE)
rng = np.random.default_rng(5)


def road(n, gain):
    if gain <= 0:
        return np.zeros(n, np.int16)
    o = int(rng.integers(0, len(noise) - n - 1))
    return (noise[o:o + n].astype(np.float32) * gain).astype(np.int16)


def mix(x, gain):
    return np.clip(x.astype(np.int32) + road(len(x), gain), -32768, 32767).astype(np.int16)


eng = E.Engine()
print(f"Whisper threads: {E.CFG['THREADS']}, wake mode: {E.CFG['WAKE_MODE']}", flush=True)


def run_stream(stream, until_case=True, tail_s=0.0):
    """Feed in real time like a microphone. Stops early once a case has started (plus 1 s). Returns the Live and its events."""
    events = []
    live = E.Live(eng, lambda ev: events.append(ev))
    chunk = SR // 10
    t0 = time.time()
    for i in range(0, len(stream), chunk):
        live.feed(stream[i:i + chunk].tobytes())
        time.sleep(max(0.0, t0 + (i + chunk) / SR - time.time()))
        if until_case and any(e["type"] == "case" for e in events):
            for _ in range(10):
                time.sleep(0.1)
            break
    for _ in range(100):
        if not live.pending():
            break
        time.sleep(0.2)
    live.close()
    return live, events


def drop_case(events):
    for e in events:
        if e["type"] == "case" and e.get("state") == "started":
            shutil.rmtree(E.SESSIONS / e["sid"], ignore_errors=True)


def summarize(label, results):
    n = len(results)
    wk = sum(1 for r in results if r["wake"])
    st = sum(1 for r in results if r["started"])
    tags = [r["tag_s"] for r in results if r["tag_s"] is not None]
    wd = [r["wake_after_s"] for r in results if r["wake_after_s"] is not None]
    print(f"{label:34s} wake heard {wk}/{n}   case started by voice {st}/{n}"
          + (f"   wake {np.median(wd):+.1f}s after the clip began (median)" if wd else "")
          + (f"   tag {np.median(tags):.1f}s median, {max(tags):.1f}s slowest (after you stop talking)" if tags else ""), flush=True)


def one(clip_audio, gain, clip_len_s):
    lead = 2.0
    tail = E.CFG["AWAKE_SECONDS"] + 2.0
    stream = np.concatenate([road(int(lead * SR), gain), mix(clip_audio, gain), road(int(tail * SR), gain)])
    live, ev = run_stream(stream)
    w = [e for e in ev if e["type"] == "wake"]
    cmd = [e for e in ev if e["type"] == "command" and e["name"] == "start_recording" and e["source"] == "voice"]
    r = {"wake": bool(w), "started": bool(cmd), "tag_s": cmd[0]["timing"]["total_s"] if cmd and cmd[0].get("timing") else None,
         "wake_after_s": (w[0]["clock"] - lead) if w else None, "calls": live.stats["whisper_calls"], "timeouts": live.stats["awake_timeouts"]}
    drop_case(ev)
    return r


allres = {}
for gain in GAINS:
    tag = "clean" if gain == 0 else f"road noise x{gain}"
    print(f"\n=== {tag} ===", flush=True)
    res = [one(read_wav(f), gain, 0) for (_, f, _) in one_breath]
    summarize("A one breath", res)
    allres[("A", gain)] = res
    res = []
    for (_, fj, _) in jarvis_alone:
        for (_, fc, _) in command_only[:2]:
            a = np.concatenate([read_wav(fj), np.zeros(SR, np.int16), read_wav(fc)])
            res.append(one(a, gain, 0))
    summarize("B 'Hey Jarvis' + 1 s pause + command", res)
    allres[("B", gain)] = res

print("\n=== C idle talk: no wake word, nothing should reach Whisper ===", flush=True)
for gain in [g for g in GAINS if g in (0.0, 0.01)] or GAINS[:1]:
    parts = [road(3 * SR, gain)]
    for (_, f, _) in idle:
        parts += [mix(read_wav(f), gain), road(int(2.5 * SR), gain)]
    stream = np.concatenate(parts)
    cpu0, w0 = time.process_time(), time.time()
    live = E.Live(eng, lambda ev: None)
    chunk = SR // 10
    t0 = time.time()
    for i in range(0, len(stream), chunk):
        live.feed(stream[i:i + chunk].tobytes())
        time.sleep(max(0.0, t0 + (i + chunk) / SR - time.time()))
    time.sleep(1.0)
    cpu, wall = time.process_time() - cpu0, time.time() - w0
    audio = len(stream) / SR
    live.close()
    st = live.stats
    print(f"noise x{gain}: {audio:.0f} s of idle talk ({len(idle)} lines): Whisper calls {st['whisper_calls']}, lines skipped {st['idle_skipped']}, "
          f"false wakes {st['wakes']} (~{st['wakes'] * 3600 / audio:.0f} per hour of this kind of talk), "
          f"CPU used {cpu:.1f}s over {wall:.0f}s = {100 * cpu / wall:.0f}% of one core", flush=True)

print(f"\npeak memory: {limits.peak_memory_mb()} MB")
print("PC timings; a Pi 5 is slower per core (UNVERIFIED: guess x3 to x6 for Whisper), so multiply the 'tag' numbers by that range for a rough Pi expectation.")
