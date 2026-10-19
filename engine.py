"""FALLBACK_CAMOTE engine.

LIVE path  (fast, light):  mic stream -> wake word (audio) + speech cutter (Silero VAD) -> priority queue -> Whisper (run13) -> CAMOTE matchers
                           -> command / violation events.  No live transcript is shown; only what the enforcer needs (commands, violation list, timing).
AFTER-STOP path (slow, careful): the recorded video's audio -> Whisper with more care -> full time-stamped transcript -> cross-check against the live list.

Speaker recognition is NOT in here yet (owner: test first, embedding later). Every line is labelled "-" and the hook is `speaker_of()`.
The matchers (analyze, violation_matcher, ...) are the CAMOTE ones, copied unchanged into ./camote_speech.
"""
import collections
import json
import math
import os
import re
import threading
import time
import wave
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np

from camote_speech import analyze
from camote_speech import violation_matcher as vm
from camote_speech.offenses import OFFENSES, full_names
from camote_speech.transcribe import PROMPT
from vadcut import SR, StreamCutter, cut_audio

HERE = Path(__file__).resolve().parent
SESSIONS = HERE / "sessions"


def _env(name, default):
    return os.environ.get(name) or default


CFG = {
    "WHISPER_DIR": _env("FALLBACK_WHISPER_DIR", str(HERE.parent / "03_models" / "run13-ct2-int8")),
    "THREADS": int(_env("FALLBACK_THREADS", "4")),
    "LIVE_BEAM": int(_env("FALLBACK_LIVE_BEAM", "1")),          # same as the deployed handoff (fast)
    "FINAL_BEAM": int(_env("FALLBACK_FINAL_BEAM", "5")),        # after the stop there is time: look at more candidates
    "ON": float(_env("FALLBACK_VAD_ON", "0.35")),               # speech probability that counts as speech (0.25 for far voices)
    "END_SILENCE": float(_env("FALLBACK_END_SILENCE", "0.7")),  # conversation lines: this much quiet ends a line
    "CAP": float(_env("FALLBACK_CAP", "12")),
    "AWAKE_SECONDS": float(_env("FALLBACK_AWAKE_SECONDS", "8")),
    "AWAKE_END_SILENCE": float(_env("FALLBACK_AWAKE_END_SILENCE", "0.45")),  # right after "Hey Jarvis" the next line is a short command
    "AWAKE_CAP": float(_env("FALLBACK_AWAKE_CAP", "6")),
    # wake word. "both" = audio model OR the word "Jarvis" heard in the text (measured: the audio model alone caught only 1 of 8 of the owner's "Jarvis" clips);
    # "audio" = audio model only (idle speech is never sent to Whisper: lightest); "off" = every idle line may carry a command
    "WAKE_MODE": _env("FALLBACK_WAKE_MODE", "both"),
    "WAKE_THRESHOLD": float(_env("FALLBACK_WAKE_THRESHOLD", "0.3")),      # 0.3 on the owner's clips: 0 false wakes in 252 other clips and 18.5 min of road noise
    "JOIN_LINES": _env("FALLBACK_JOIN_LINES", "1") == "1",      # also read two neighbouring lines together (a violation split by a pause)
    "MERGE_GAP": 1.5,                                            # short lines closer than this (seconds) go to Whisper as ONE call
    "MERGE_MAX": 12.0,
    "MAX_WAIT": 8.0,                                             # a line that waited longer than this is flagged "late"
}
COMMANDS = ["start_recording", "scan_license", "scan_cr", "capture", "end_recording"]
LOW_VALUE_LABELS = ("License", "Registration", "Documents")


# ---------------------------------------------------------------------------------------------------------- text helpers
def speaker_of(_pcm):
    """Hook for speaker embedding (enforcer / driver). Not enabled yet: everything is "-"."""
    return "-"


def looks_like_loop(text):
    """Whisper sometimes loops on noise ("satang satang satang..."). Such text is thrown away, never matched."""
    words = text.split()
    if len(words) < 6:
        return False
    raw = text.encode("utf-8")
    if len(raw) / max(1, len(zlib.compress(raw))) > 2.4:
        return True
    for n in (1, 2, 3):
        grams = [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]
        top = max(collections.Counter(grams).values())
        if top >= 4 and top * n >= 0.6 * len(words):
            return True
    return False


_WAKE_SOUNDS = None


def wake_in_text(text, r_wake_text=False):
    """The wake phrase in the TEXT. The CAMOTE check first, then a looser one: a token near the start or end of the line (or two tokens glued
    together) that sounds like 'jarvis' ("Garbis record", "Gard peke"). Measured on the owner's Set 3 clips: strict 7/8, loose 8/8, 0 false in 252 other lines."""
    global _WAKE_SOUNDS
    if r_wake_text:
        return True
    from camote_speech.command_matcher import _lev
    from camote_speech.phonetic_fix import filipino_metaphone as fp
    if _WAKE_SOUNDS is None:
        _WAKE_SOUNDS = [fp(w) for w in ("jarvis", "harvis", "yarbis", "garvis", "charvis", "jarpis", "jarbis")]
    toks = re.findall(r"[a-z']+", (text or "").lower())
    cands = set(toks[:3] + toks[-3:])
    for i in range(len(toks) - 1):
        if i < 3 or i >= len(toks) - 3:
            cands.add(toks[i] + toks[i + 1])
    for c in cands:
        f = fp(c)
        if len(f) >= 4 and any(_lev(f, t) <= 2 for t in _WAKE_SOUNDS):
            return True
    return False


_DENY_NEXT = re.compile(r"\b(?:hindi totoo|mali yun|mali yan)\b")


def _denies(text):
    low = " ".join(vm._toks(text))
    return bool(vm.DENIAL.search(low) or _DENY_NEXT.search(low))


def session_violations(turns, join=True, max_gap=2.5):
    """Violation labels over a whole conversation. turns: [{t0,t1,text,violations}] in spoken order.
    A line does not count when the very next line flatly denies it ("Hindi po ako lasing")."""
    found = []
    for i, t in enumerate(turns):
        nxt = turns[i + 1] if i + 1 < len(turns) else None
        if nxt is not None and _denies(nxt["text"]):
            continue
        for lab in t.get("violations", []):
            if lab not in found:
                found.append(lab)
    if join:
        for a, b in zip(turns, turns[1:]):
            if b["t0"] - a["t1"] > max_gap:
                continue
            joined = a["text"] + " " + b["text"]
            words = set(re.findall(r"[a-z']+", joined.lower()))
            for lab in vm.detect(joined)["violations"]:
                if lab in found or lab in a.get("violations", []) or lab in b.get("violations", []):
                    continue
                if lab in LOW_VALUE_LABELS and not (words & vm.NEG):
                    continue
                found.append(lab)
    return found


# ---------------------------------------------------------------------------------------------------------- models
class Whisper:
    def __init__(self, model_dir, threads):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(str(model_dir), device="cpu", compute_type="int8", cpu_threads=threads)
        self.lock = threading.Lock()

    def transcribe(self, pcm, beam=1):
        x = np.asarray(pcm).astype(np.float32) / 32768.0
        cap = min(int(math.ceil(len(x) / SR * 12)) + 10, 200)
        with self.lock:
            segs, _ = self.model.transcribe(x, language="tl", beam_size=beam, vad_filter=False, condition_on_previous_text=False,
                                            initial_prompt=PROMPT, temperature=0.0, max_new_tokens=cap)
            text = " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6)
        return text.strip()


class WakeDetector:
    """'Hey Jarvis' on the AUDIO (openWakeWord, the same stand-in model as the handoff). Cooldown runs on the audio clock."""

    def __init__(self, threshold=0.3):
        from openwakeword.model import Model
        self.model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        self.last = -99.0
        self.threshold = threshold

    def feed(self, frame, clock):
        scores = self.model.predict(frame)
        score = max(scores.values()) if scores else 0.0
        if score > self.threshold and clock - self.last > 3.0:
            self.last = clock
            return True
        return False


class Engine:
    def __init__(self):
        d = CFG["WHISPER_DIR"]
        if not os.path.isdir(d):
            raise SystemExit(f"Whisper model folder not found: {d}\nSet FALLBACK_WHISPER_DIR to a CTranslate2 model folder (run13-ct2-int8).")
        print(f"[fallback] loading Whisper: {d}", flush=True)
        t0 = time.time()
        self.whisper = Whisper(d, CFG["THREADS"])
        self.whisper.transcribe(np.zeros(SR, np.int16))      # warm up
        analyze("warm up")                                    # loads the phonetic database now, not on the first sentence
        print(f"[fallback] Whisper ready ({time.time() - t0:.1f}s)", flush=True)
        try:
            WakeDetector()
            self.wake_available = True
        except Exception as e:
            print(f"[fallback] wake word unavailable ({e!r}); the text wake backup is used", flush=True)
            self.wake_available = False
        self.lives = set()
        SESSIONS.mkdir(exist_ok=True)

    def live_busy(self):
        return any(l.pending() for l in list(self.lives))


# ---------------------------------------------------------------------------------------------------------- LIVE path
class Live:
    """One listening connection: audio in -> events out. Everything slow happens in the worker thread."""

    def __init__(self, engine, emit_cb, **over):
        self.e = engine
        self.cb = emit_cb
        self.cfg = {**CFG, **over}
        c = self.cfg
        self.cutter = StreamCutter(on=c["ON"], end_silence=c["END_SILENCE"], cap=c["CAP"])
        self.wake = WakeDetector(c["WAKE_THRESHOLD"]) if engine.wake_available and c["WAKE_MODE"] != "off" else None
        self.store = bytearray()
        self.samples = 0
        self.wtail = np.zeros(0, np.int16)
        self.recording = False
        self.wake_at = None
        self.awake_until = -1.0
        self.history, self.live_labels = [], []
        self.case = None
        self.markers = {}
        self.case_events = []
        self.ignored = 0
        self._in_speech = False
        self._short_mode = None
        self.cv = threading.Condition()
        self.jobs, self.busy, self.stopped, self.seq = [], False, False, 0
        threading.Thread(target=self._worker, daemon=True).start()
        engine.lives.add(self)

    # ------------------------------------------------------------------ small helpers
    @property
    def clock(self):
        return self.samples / SR

    def pending(self):
        with self.cv:
            return len(self.jobs) + (1 if self.busy else 0)

    def close(self):
        with self.cv:
            self.stopped = True
            self.cv.notify_all()
        self.e.lives.discard(self)

    def _emit(self, ev):
        ev["clock"] = round(self.clock, 2)
        ev["time"] = datetime.now().strftime("%H:%M:%S")
        if self.recording or ev.get("type") in ("case",):
            self.case_events.append({k: v for k, v in ev.items()})
        self.cb(ev)

    def settings(self, **kw):
        if "JOIN_LINES" in kw:
            self.cfg["JOIN_LINES"] = bool(kw["JOIN_LINES"])
        if kw.get("WAKE_MODE") in ("both", "audio", "off"):
            self.cfg["WAKE_MODE"] = kw["WAKE_MODE"]
            if self.wake is None and kw["WAKE_MODE"] != "off" and self.e.wake_available:
                self.wake = WakeDetector(self.cfg["WAKE_THRESHOLD"])
        if "ON" in kw:
            self.cfg["ON"] = self.cutter.on = float(kw["ON"])

    # ------------------------------------------------------------------ audio in (called from the web thread; must stay light)
    def feed(self, data):
        pcm = np.frombuffer(data, dtype=np.int16)
        if len(pcm) == 0:
            return
        self.store.extend(data)
        self.samples += len(pcm)
        self._wake(pcm)
        self._mode()
        for seg in self.cutter.feed(pcm):
            self._on_segment(seg)
        if self.cutter.in_seg != self._in_speech:
            self._in_speech = self.cutter.in_seg
            self.cb({"type": "vad", "speech": self._in_speech, "clock": round(self.clock, 2)})

    def _wake(self, pcm):
        self.wtail = np.concatenate([self.wtail, pcm])
        while len(self.wtail) >= 1280:
            frame, self.wtail = self.wtail[:1280], self.wtail[1280:]
            if self.wake is not None and self.wake.feed(frame, self.clock):
                self._set_wake("audio")

    def _set_wake(self, source):
        self.wake_at = self.clock
        self.awake_until = self.clock + self.cfg["AWAKE_SECONDS"]
        self._emit({"type": "wake", "source": source, "awake_seconds": self.cfg["AWAKE_SECONDS"]})

    def _mode(self):
        """Right after the wake word the next line is a short command: end it sooner. During a case lines are conversation: normal limits."""
        short = (not self.recording) and self.clock <= self.awake_until
        if short != self._short_mode:
            self._short_mode = short
            c = self.cfg
            self.cutter.set_limits(end_silence=c["AWAKE_END_SILENCE"] if short else c["END_SILENCE"], cap=c["AWAKE_CAP"] if short else c["CAP"])

    # ------------------------------------------------------------------ a finished line -> queue
    def _on_segment(self, seg):
        awake = self.wake_at is not None and seg["t1"] >= self.wake_at - 0.5 and seg["t0"] <= self.awake_until
        mode = self.cfg["WAKE_MODE"]
        if awake or (not self.recording and mode == "off"):
            prio = 0                                        # a command is expected: first in line
        elif self.recording:
            prio = 1                                        # a conversation line of the case
        elif mode == "both":
            prio = 2                                        # idle speech: heard last; only counts if the text itself contains the wake phrase
        else:
            self.ignored += 1
            self.cb({"type": "segment", "action": "ignored", "dur": round(len(seg["pcm"]) / SR, 2), "ignored_total": self.ignored,
                     "clock": round(self.clock, 2)})
            return
        self.seq += 1
        seg.update(prio=prio, enq=time.time(), seq=self.seq, merged=1)
        with self.cv:
            self.jobs.append(seg)
            self.cv.notify()
        self.cb({"type": "segment", "action": "queued", "prio": prio, "dur": round(len(seg["pcm"]) / SR, 2), "reason": seg["reason"],
                 "queue": len(self.jobs), "clock": round(self.clock, 2)})

    # ------------------------------------------------------------------ worker
    def _take(self):
        with self.cv:
            while not self.jobs and not self.stopped:
                self.cv.wait(0.5)
            if self.stopped:
                return None
            self.jobs.sort(key=lambda j: (j["prio"], j["seq"]))
            first = self.jobs.pop(0)
            self.busy = True
            if first["prio"] == 1:
                c = self.cfg
                group = [first]
                for j in sorted([j for j in self.jobs if j["prio"] == 1], key=lambda j: j["t0"]):
                    if j["t0"] < first["t0"]:
                        continue
                    if j["t0"] - group[-1]["t1"] <= c["MERGE_GAP"] and j["t1"] - first["t0"] <= c["MERGE_MAX"]:
                        group.append(j)
                        self.jobs.remove(j)
                if len(group) > 1:
                    parts = [group[0]["pcm"]]
                    for a, b in zip(group, group[1:]):
                        parts += [np.zeros(int(min(max(b["t0"] - a["t1"], 0), 0.4) * SR), np.int16), b["pcm"]]
                    last = group[-1]
                    first = {**first, "pcm": np.concatenate(parts), "t1": last["t1"], "speech_t1": last["speech_t1"], "detected_t": last["detected_t"],
                             "wait_s": last["wait_s"], "merged": len(group), "enq": min(g["enq"] for g in group)}
            return first

    def _worker(self):
        while True:
            job = self._take()
            if job is None:
                return
            try:
                self._handle(job)
            except Exception as ex:
                self._emit({"type": "error", "message": repr(ex)})
            finally:
                with self.cv:
                    self.busy = False

    def _handle(self, job):
        queue_s = time.time() - job["enq"]
        t0 = time.time()
        text = self.e.whisper.transcribe(job["pcm"], beam=self.cfg["LIVE_BEAM"])
        whisper_s = time.time() - t0
        base = {"audio_s": round(len(job["pcm"]) / SR, 2), "wait_s": round(job["wait_s"], 2), "queue_s": round(queue_s, 2),
                "whisper_s": round(whisper_s, 2), "merged": job["merged"], "late": queue_s > self.cfg["MAX_WAIT"], "prio": job["prio"],
                "reason": job["reason"]}
        if not text or looks_like_loop(text):
            self._emit({"type": "line", "text": "", "dropped": "loop" if text else "empty", "timing": {**base, "total_s": round(base["wait_s"] + queue_s + whisper_s, 2)}})
            return
        t1 = time.time()
        r = analyze(text)
        match_s = time.time() - t1
        timing = {**base, "match_s": round(match_s, 4), "total_s": round(job["wait_s"] + queue_s + whisper_s + match_s, 2)}
        has_wake = wake_in_text(r["raw"], r["wake_text"])
        if job["prio"] == 2 and not has_wake:
            self._emit({"type": "line", "text": "", "heard": r["text"], "dropped": "no wake word", "timing": timing})   # idle talk: not for us, nothing is kept
            return
        if has_wake and (self.wake_at is None or self.clock > self.awake_until):
            self._set_wake("text")                            # backup: the wake phrase was in the text but the audio model missed it
        self._emit({"type": "line", "speaker": speaker_of(job["pcm"]), "raw": r["raw"], "text": r["text"], "commands": r["commands"],
                    "violations": r["violations"], "hints": r["hints"], "timing": timing})
        if os.environ.get("FALLBACK_LOG", "1") == "1":                # the terminal shows what the live path heard (for testing; nothing is shown in the app)
            print(f"[live {self.clock:6.1f}s] {r['text'][:70]!r} cmds={r['commands']} viol={r['violations']} ({timing['total_s']}s)", flush=True)
        line = {"t0": job["t0"], "t1": job["t1"], "text": r["text"], "violations": r["violations"], "commands": r["commands"]}
        if self.recording:
            self.history.append(line)
        for name in r["commands"]:
            self.do_command(name, "voice", job, timing, r["text"])
        if self.recording:
            self._update_violations(timing)

    # ------------------------------------------------------------------ commands + case
    def do_command(self, name, source, job=None, timing=None, text=""):
        if name not in COMMANDS:
            return
        clock = self.clock
        ev = {"type": "command", "name": name, "source": source, "timing": timing, "text": text}
        if name == "start_recording":
            if self.recording:
                self._emit({**ev, "note": "already recording"})
                return
            self._emit(ev)
            self.recording = True
            self.history, self.live_labels, self.case_events = [], [], []
            sid = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.case = {"sid": sid, "start": job["t0"] if job else clock, "end": None}
            self._emit({"type": "case", "state": "started", "sid": sid, "start": round(self.case["start"], 2)})
        elif name == "end_recording":
            if not self.recording:
                self._emit({**ev, "note": "not recording"})
                return
            self._emit(ev)
            self._update_violations(timing)
            self.recording = False
            self.case["end"] = job["t1"] if job else clock
            self._save_case()
            self._emit({"type": "case", "state": "ended", "sid": self.case["sid"], "start": round(self.case["start"], 2), "end": round(self.case["end"], 2),
                        "violations": self.live_labels, "violation_names": full_names(self.live_labels), "lines": len(self.history)})
        else:
            self._emit(ev)
        self.awake_until = -1.0                             # one wake word = one command line
        self._short_mode = None

    def mark(self, vid):
        self.markers[vid] = round(self.clock, 3)

    def _update_violations(self, timing):
        now = session_violations(self.history, join=self.cfg["JOIN_LINES"])
        if now != self.live_labels:
            new = [x for x in now if x not in self.live_labels]
            self.live_labels = now
            self._emit({"type": "violations", "labels": now, "names": full_names(now), "new": new, "timing": timing})

    def _save_case(self):
        sid = self.case["sid"]
        d = SESSIONS / sid
        d.mkdir(parents=True, exist_ok=True)
        pre, post = 5.0, 3.0
        a = max(0.0, self.case["start"] - pre)
        b = min(self.clock, self.case["end"] + post)
        pcm = np.frombuffer(bytes(self.store), dtype=np.int16)[int(a * SR):int(b * SR)]
        with wave.open(str(d / "stream_case.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes(pcm.tobytes())
        meta = {"sid": sid, "start": self.case["start"], "end": self.case["end"], "stream_offset": a, "markers": dict(self.markers),
                "live_labels": self.live_labels, "live_names": full_names(self.live_labels), "live_lines": self.history,
                "live_events": self.case_events, "config": {k: v for k, v in self.cfg.items()}}
        (d / "case.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------------------------------------------- AFTER-STOP path
def load_audio_any(path):
    """Audio track of a video/audio file (webm, mp4, wav...) as 16 kHz mono int16, via PyAV (no ffmpeg install needed)."""
    import av
    out = []
    with av.open(str(path)) as c:
        st = next((s for s in c.streams if s.type == "audio"), None)
        if st is None:
            raise ValueError("the file has no audio track")
        rs = av.AudioResampler(format="s16", layout="mono", rate=SR)
        for fr in c.decode(st):
            for f in rs.resample(fr):
                out.append(f.to_ndarray().reshape(-1))
        for f in rs.resample(None) or []:
            out.append(f.to_ndarray().reshape(-1))
    return np.concatenate(out).astype(np.int16) if out else np.zeros(0, np.int16)


def _read_wav(path):
    with wave.open(str(path), "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def find_video(d):
    for n in ("video.webm", "video.mp4", "video.mkv", "video.mov"):
        if (d / n).exists():
            return d / n
    return None


def remux_with_duration(src, dst):
    """Copy the streams of a MediaRecorder file into a new file that carries its duration and seek index. True when it worked."""
    import av
    try:
        i = av.open(str(src))
        o = av.open(str(dst), "w", format=dst.suffix.lstrip(".") if dst.suffix == ".webm" else "mp4")
        m = {s.index: o.add_stream_from_template(s) for s in i.streams if s.type in ("video", "audio")}
        for p in i.demux():
            if (p.dts is None and p.pts is None) or p.stream.index not in m:
                continue
            p.stream = m[p.stream.index]
            o.mux(p)
        o.close()
        i.close()
        return dst.stat().st_size > 0
    except Exception as ex:
        print("[fallback] remux skipped:", repr(ex), flush=True)
        try:
            dst.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def run_final(engine, sid, source, progress=None, line_cb=None, beam=None):
    """Careful pass over everything that was said. source: 'video' (the recorded video's audio) or 'stream' (the audio the live path heard).
    Waits for the live path between calls, so a new stop is never slowed down by this."""
    d = SESSIONS / sid
    meta = json.loads((d / "case.json").read_text(encoding="utf-8"))
    beam = beam or CFG["FINAL_BEAM"]
    t_all = time.time()
    if source == "video":
        vp = find_video(d)
        if vp is None:
            raise ValueError("no video uploaded for this case")
        pcm = load_audio_any(vp)
        vid = meta.get("video_vid")
        origin = meta["markers"].get(vid) if vid else None
        cs, ce = (meta["start"] - origin, meta["end"] - origin) if origin is not None else (None, None)
    else:
        pcm = _read_wav(d / "stream_case.wav")
        origin = meta["stream_offset"]
        cs, ce = meta["start"] - origin, meta["end"] - origin
    audio_s = len(pcm) / SR
    segs = cut_audio(pcm, on=CFG["ON"], end_silence=CFG["END_SILENCE"], cap=24.0, soft_margin=8.0)
    lines = []
    for i, s in enumerate(segs):
        while engine.live_busy():
            time.sleep(0.05)
        t0 = time.time()
        text = engine.whisper.transcribe(s["pcm"], beam=beam)
        dt = time.time() - t0
        if text and not looks_like_loop(text):
            r = analyze(text)
            in_case = cs is None or (s["t1"] >= cs - 1.0 and s["t0"] <= ce + 1.0)
            line = {"t0": round(s["t0"], 2), "t1": round(s["t1"], 2), "speaker": speaker_of(s["pcm"]), "raw": r["raw"], "text": r["text"],
                    "commands": r["commands"], "violations": r["violations"], "hints": r["hints"], "in_case": in_case, "seconds": round(dt, 2)}
            lines.append(line)
            if line_cb:
                line_cb(line)
        if progress:
            progress(i + 1, len(segs))
    history = [l for l in lines if l["in_case"]]
    final = session_violations(history, join=CFG["JOIN_LINES"])
    live = meta.get("live_labels", [])
    report = {"sid": sid, "source": source, "beam": beam, "audio_s": round(audio_s, 1), "seconds": round(time.time() - t_all, 1),
              "segments": len(segs), "lines": lines, "violations": final, "violation_names": full_names(final),
              "live_violations": live, "live_names": full_names(live),
              "confirmed": [x for x in live if x in final], "live_only": [x for x in live if x not in final], "final_only": [x for x in final if x not in live],
              "case_start": cs, "case_end": ce}
    report["rtf"] = round(report["seconds"] / max(0.1, audio_s), 2)
    (d / f"final_{source}.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    return report


def list_sessions():
    out = []
    for d in sorted(SESSIONS.iterdir(), reverse=True) if SESSIONS.exists() else []:
        p = d / "case.json"
        if p.exists():
            m = json.loads(p.read_text(encoding="utf-8"))
            out.append({"sid": d.name, "live_names": m.get("live_names", []), "video": find_video(d) is not None,
                        "final": [f.stem.replace("final_", "") for f in d.glob("final_*.json")]})
    return out


def offense_list():
    return [{"label": k, "code": v[0], "name": v[1], "full": full_names([k])[0]} for k, v in OFFENSES.items()]
