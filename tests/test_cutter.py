"""Checks the speech cutter: streaming result must match the whole-file result, pure road noise must give 0 segments, and real clips must be cut cleanly.
Run:  <python> tests\\test_cutter.py        (read-only on the recordings and on Bgdataset-1.wav; writes nothing)
"""
import csv
import glob
import os
import random
import sys
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from vadcut import SR, StreamCutter, cut_audio  # noqa: E402

CAMOTE = os.path.abspath(os.path.join(HERE, "..", ".."))
S04 = os.path.join(CAMOTE, "01_inbox", "owner_sessions", "S04_20261017")
NOISE = os.path.join(CAMOTE, "Bgdataset-1.wav")


def read_wav(p, seconds=None):
    with wave.open(p) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        n = w.getnframes() if seconds is None else min(w.getnframes(), int(seconds * rate))
        x = np.frombuffer(w.readframes(n), dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]
    if rate != SR:
        from scipy.signal import resample_poly
        from math import gcd
        g = gcd(rate, SR)
        x = resample_poly(x.astype(np.float32), SR // g, rate // g).astype(np.int16)
    return x


ok = True


def check(name, cond, info=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + (f"   {info}" if info else ""), flush=True)


# 1. pure road noise: 0 segments (the old loudness cutter made 73 per 18 minutes at the 15 s cap)
noise = read_wav(NOISE)
segs = cut_audio(noise)
check("pure road noise gives 0 segments", len(segs) == 0, f"{len(segs)} segments in {len(noise) / SR / 60:.1f} min")

# 2. real clips between noise: streaming in odd chunk sizes equals one big call
rows = list(csv.DictReader(open(os.path.join(S04, "labels.csv"), encoding="utf-8")))
random.seed(3)
random.shuffle(rows)
parts, truth, pos = [], [], 0
rng = np.random.default_rng(3)
noise_lvl = 0.0
for r in rows[:12]:
    w = sorted(glob.glob(os.path.join(S04, r["line_id"] + "_*.wav")))
    if not w:
        continue
    x = read_wav(w[-1])
    gap = int(random.uniform(2.5, 4.5) * SR)
    off = int(rng.integers(0, len(noise) - gap - 1))
    bg = noise[off:off + gap].astype(np.float32) * 0.15          # road noise about 15% of its recorded level: a quiet-ish road
    parts.append(bg.astype(np.int16))
    base = sum(len(p) for p in parts)
    nfr = len(x) // 512                                            # the clip has its own silence at both ends: truth = the loud part only
    lv = np.abs(x[:nfr * 512].astype(np.float32)).reshape(nfr, 512).mean(axis=1)
    act = np.where(lv > 0.25 * lv.max())[0]
    truth.append((base + int(act[0]) * 512, base + (int(act[-1]) + 1) * 512))
    parts.append(x)
stream = np.concatenate(parts + [np.zeros(SR * 2, np.int16)])
whole = cut_audio(stream)
c = StreamCutter()
live, i = [], 0
sizes = [1600, 777, 4096, 100, 3200]
k = 0
while i < len(stream):
    n = sizes[k % len(sizes)]
    live += c.feed(stream[i:i + n])
    i += n
    k += 1
live += c.flush()
check("live (odd chunk sizes) finds as many lines as the whole-file call", abs(len(live) - len(whole)) <= 1, f"live {len(live)}, whole {len(whole)}, spoken {len(truth)}")
hit = 0
for a, b in truth:
    cov = sum(max(0, min(b, int(s["t1"] * SR)) - max(a, int(s["t0"] * SR))) for s in live) / (b - a)
    hit += cov >= 0.8
check("every spoken line is covered (>= 80%)", hit == len(truth), f"{hit}/{len(truth)}")
lens = [(s["t1"] - s["t0"]) for s in live]
check("no line runs to the 12 s cap", max(lens) < 11.9, f"longest {max(lens):.1f}s")
check("end-of-line wait is about 0.7 s", all(0.6 <= s["wait_s"] <= 0.85 for s in live if s["reason"] == "silence"),
      f"waits {sorted(set(round(s['wait_s'], 2) for s in live))[:4]}")

# 3. the awake (command) limits end a line sooner
c = StreamCutter()
c.set_limits(end_silence=0.45, cap=6.0)
sh = c.feed(stream) + c.flush()
check("command mode ends lines sooner (0.45 s wait)", all(s["wait_s"] <= 0.6 for s in sh if s["reason"] == "silence"), f"{len(sh)} lines")

print("\nALL PASS" if ok else "\nSOME FAILED")
sys.exit(0 if ok else 1)
