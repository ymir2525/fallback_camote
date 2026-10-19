"""Builds samples/sample_conversation.wav: a short stop made of the owner's own Set 3 clips (enforcer + driver) over very light road noise.
Used by the "Play sample conversation" button so the app can be tried without a microphone. Reads recordings, writes only into FALLBACK_CAMOTE\\samples.
Run:  <python> tests\\make_sample.py
"""
import csv
import glob
import os
import sys
import wave
from math import gcd

import numpy as np
from scipy.signal import resample_poly

HERE = os.path.dirname(os.path.abspath(__file__))
CAMOTE = os.path.abspath(os.path.join(HERE, "..", ".."))
S01 = os.path.join(CAMOTE, "01_inbox", "owner_sessions", "S01_20261018")
NOISE = os.path.join(CAMOTE, "Bgdataset-1.wav")
OUT = os.path.join(os.path.dirname(HERE), "samples", "sample_conversation.wav")
SR = 16000


def read_wav(p):
    with wave.open(p) as w:
        rate, ch = w.getframerate(), w.getnchannels()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    if ch > 1:
        x = x.reshape(-1, ch)[:, 0]
    if rate != SR:
        g = gcd(rate, SR)
        x = resample_poly(x.astype(np.float32), SR // g, rate // g).astype(np.int16)
    return x


rows = {r["line_id"]: r for r in csv.DictReader(open(os.path.join(S01, "labels.csv"), encoding="utf-8"))}
drivers = [lid for lid, r in rows.items() if r["part"] == "R3" and r["role"] == "DRIVER"][:2]
ORDER = ["R2001",          # idle talk before the wake word: must be ignored
         "R1005",          # "Hey Jarvis, mag-record ka."   -> start the case
         "R2005", drivers[0], "R2010", drivers[1],
         "R1017",          # "Scan license na."
         "R1033",          # "Capture mo yung plate."
         "R1043"]          # "Kaso tapos na, Jarvis."        -> end the case
noise = read_wav(NOISE)
rng = np.random.default_rng(21)


def road(n):
    o = int(rng.integers(0, len(noise) - n - 1))
    return (noise[o:o + n].astype(np.float32) * 0.01).astype(np.int16)


parts = [road(SR * 3)]
for lid in ORDER:
    f = sorted(glob.glob(os.path.join(S01, lid + "_*.wav")))[-1]
    x = read_wav(f)
    parts += [np.clip(x.astype(np.int32) + road(len(x)), -32768, 32767).astype(np.int16), road(int(SR * 2.6))]
out = np.concatenate(parts)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with wave.open(OUT, "wb") as w:
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(out.tobytes())
print("wrote", OUT, f"{len(out) / SR:.0f} s")
for lid in ORDER:
    print("  ", lid, rows[lid]["role"], rows[lid]["text"])
sys.exit(0)
