"""End to end, no browser: your real Set 3 clips + road noise are played into the live path IN REAL TIME, like a microphone.
Checks: wake word on real "Hey Jarvis" audio, the 5 commands, violation tags, ignored lines (no wake word, not recording), timing, and the after-stop pass.
Run:  <python> tests\\e2e_live.py            (about 2 minutes; reads recordings, writes only to FALLBACK_CAMOTE\\sessions and removes it again)
"""
import csv
import glob
import json
import os
import shutil
import sys
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import engine as E  # noqa: E402
from camote_speech import analyze  # noqa: E402

CAMOTE = os.path.abspath(os.path.join(HERE, "..", ".."))
S01 = os.path.join(CAMOTE, "01_inbox", "owner_sessions", "S01_20261018")
NOISE = os.path.join(CAMOTE, "Bgdataset-1.wav")
SR = 16000
NOISE_GAIN = float(os.environ.get("E2E_NOISE_GAIN", "0.12"))


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


def clip(line_id):
    f = sorted(glob.glob(os.path.join(S01, line_id + "_*.wav")))[-1]
    return read_wav(f), rows[line_id]["text"], rows[line_id]["role"]


def pick_violation_lines(n):
    out = []
    for lid, r in rows.items():
        if r["part"] == "R2" and r["role"] == "OFFICER":
            a = analyze(r["text"])
            if a["violations"] and not a["commands"] and len(out) < n:
                if all(set(a["violations"]) != set(analyze(rows[o]["text"])["violations"]) for o in out):
                    out.append(lid)
    return out


V = pick_violation_lines(4)
driver = [lid for lid, r in rows.items() if r["part"] == "R3" and r["role"] == "DRIVER"][:2]
SCRIPT = [  # (line id, what should happen)
    ("R2" , None),  # placeholder, replaced below
]
SCRIPT = [(V[0], "ignored"),                  # idle, no wake word: must be ignored (no Whisper call)
          ("R1005", "start_recording"),       # "Hey Jarvis, mag-record ka."  wake word + command in one breath
          (V[1], "violation"), (driver[0], "driver"), (V[2], "violation"), (driver[1], "driver"),
          ("R1017", "scan_license"),          # mid-case, no wake word needed
          ("R1033", "capture"),
          (V[3], "violation"),
          ("R1043", "end_recording")]         # "Kaso tapos na, Jarvis."
noise = read_wav(NOISE)
rng = np.random.default_rng(11)


def road(seconds=None, n=None):
    n = n if n is not None else int(round(seconds * SR))
    o = int(rng.integers(0, len(noise) - n - 1))
    return (noise[o:o + n].astype(np.float32) * NOISE_GAIN).astype(np.int16)


parts, marks = [road(5.0)], []
pos = len(parts[0])
for lid, expect in SCRIPT:
    x, text, role = clip(lid)
    bg = road(n=len(x))
    mix = np.clip(x.astype(np.int32) + bg, -32768, 32767).astype(np.int16)
    marks.append({"line": lid, "expect": expect, "text": text, "start": pos / SR, "end": (pos + len(x)) / SR})
    parts.append(mix)
    pos += len(x)
    gap = road(3.0)
    parts.append(gap)
    pos += len(gap)
parts.append(road(4.0))
stream = np.concatenate(parts)
print(f"stream: {len(stream) / SR:.0f} s, {len(SCRIPT)} lines, noise gain {NOISE_GAIN}", flush=True)

eng = E.Engine()
events = []
live = E.Live(eng, lambda ev: events.append(ev))
live.mark("v1")
t0 = time.time()
chunk = SR // 10
for i in range(0, len(stream), chunk):
    live.feed(stream[i:i + chunk].tobytes())
    target = t0 + (i + chunk) / SR
    time.sleep(max(0.0, target - time.time()))
for _ in range(120):                      # let the queue finish
    if not live.pending():
        break
    time.sleep(0.5)
time.sleep(0.5)

ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"   {info}" if info else ""), flush=True)


wakes = [e for e in events if e["type"] == "wake"]
cmds = [e for e in events if e["type"] == "command"]
lines = [e for e in events if e["type"] == "line"]
viol = [e for e in events if e["type"] == "violations"]
segs = [e for e in events if e["type"] == "segment"]
ign = [e for e in segs if e["action"] == "ignored"]
print(f"\nwake events: {[(w['source'], w['clock']) for w in wakes]}")
print("commands:", [(c['name'], c['source'], c['timing'] and c['timing']['total_s']) for c in cmds])
for l in lines:
    t = l["timing"]
    print(f"  line @{l['clock']:6.1f}s  wait {t['wait_s']:.2f} + queue {t['queue_s']:.2f} + whisper {t['whisper_s']:.2f} = {t['total_s']:.2f}s  | {(l.get('text') or ('(dropped: ' + str(l.get('dropped')) + ') ' + str(l.get('heard', ''))))[:80]}  cmds={l.get('commands')} viol={l.get('violations')}")

want_cmds = ["start_recording", "scan_license", "capture", "end_recording"]
got_cmds = [c["name"] for c in cmds]
check("all 4 commands heard, in order", got_cmds == want_cmds, f"got {got_cmds}")
check("wake word fired on real 'Hey Jarvis' audio", any(w["source"] == "audio" for w in wakes), f"{[w['source'] for w in wakes]}")
idle_drop = [l for l in lines if l.get("dropped") == "no wake word"]
check("idle talk without the wake word is dropped (nothing kept, no command)", len(idle_drop) >= 1 or len(ign) >= 1, f"dropped {len(idle_drop)}, ignored {len(ign)}")
case = [e for e in events if e["type"] == "case"]
check("case started and ended", [c["state"] for c in case] == ["started", "ended"], f"{[c['state'] for c in case]}")

want_v = []
for m in marks:
    if m["expect"] == "violation":
        for lab in analyze(m["text"])["violations"]:
            if lab not in want_v:
                want_v.append(lab)
got_v = case[-1]["violations"] if case else []
missing = [x for x in want_v if x not in got_v]
extra = [x for x in got_v if x not in want_v]
check("violation tags are real: no tag that the perfect text would not give", not extra, f"false tags {extra}")
check("at least one violation tag found from the audio", bool(got_v), f"want {want_v}, got {got_v}, recall {len(want_v) - len(missing)}/{len(want_v)} (misses are Whisper mishearings, e.g. 'zigsab' for 'zigzag')")
check("no false commands from the driver or violation lines", got_cmds == want_cmds)
tot = [c["timing"]["total_s"] for c in cmds if c.get("timing")] + [v["timing"]["total_s"] for v in viol if v.get("timing")]
print(f"\ntag timing after the line ends: min {min(tot):.2f}s  median {np.median(tot):.2f}s  max {max(tot):.2f}s   (queue waits: {[l['timing']['queue_s'] for l in lines]})")
check("nothing waited more than 8 s in the queue", all(not l["timing"]["late"] for l in lines))

# after-stop pass from the streamed audio (the same function the video upload uses)
sid = case[-1]["sid"] if case else None
if sid:
    t = time.time()
    rep = E.run_final(eng, sid, "stream")
    print(f"\nfinal pass: {rep['audio_s']} s of audio in {rep['seconds']} s (x{rep['rtf']}), {len(rep['lines'])} lines")
    for l in rep["lines"]:
        print(f"  [{l['t0']:5.1f}-{l['t1']:5.1f}] {'*' if l['in_case'] else ' '} {l['text'][:80]}")
    check("final pass confirms the live violations", not rep["live_only"], f"live {rep['live_violations']} final {rep['violations']} live_only {rep['live_only']} final_only {rep['final_only']}")
    shutil.rmtree(E.SESSIONS / sid, ignore_errors=True)
live.close()
print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
