"""Speech cutter for FALLBACK_CAMOTE: Silero VAD (speech probability per 32 ms) instead of loudness.

Why: the old cutter treated road noise as speech and ran every segment to the 15 s cap. This one only follows speech probability.
A 150 Hz high-pass is applied to the COPY the detector hears; Whisper always gets the untouched audio.
Works on a live stream (feed() in any chunk size) and on a whole recording (cut_audio()).
"""
import numpy as np
from scipy.signal import butter, sosfilt

SR = 16000
CH = 512  # Silero VAD chunk: 32 ms

_VAD = None


def vad_model():
    global _VAD
    if _VAD is None:
        from faster_whisper.vad import get_vad_model
        _VAD = get_vad_model()
    return _VAD


class StreamCutter:
    """feed(int16 samples) -> list of finished segments. Each segment is a dict:
    pcm (int16, untouched audio incl. padding), t0/t1 (audio clock, seconds), speech_t0/speech_t1, detected_t (audio clock when the cutter decided),
    wait_s (= detected_t - speech_t1: the quiet we had to wait for), reason (silence | cap | soft_cap | end)."""

    def __init__(self, on=0.35, end_silence=0.7, cap=12.0, soft_margin=2.0, pad=0.25, min_speech=0.4, hpf_hz=150, start_run=3):
        self.on = on
        self.end_silence = end_silence
        self.cap = cap
        self.soft_margin = soft_margin
        self.pad_k = int(round(pad * SR / CH))
        self.min_speech = min_speech
        self.start_run = start_run
        if hpf_hz:
            self.sos = butter(4, hpf_hz / (SR / 2), btype="highpass", output="sos")
            self.zi = np.zeros((self.sos.shape[0], 2))
        else:
            self.sos, self.zi = None, None
        self.tail = np.zeros(0, np.int16)
        self.prev = np.zeros(CH, np.float32)
        self.buf = np.zeros(0, np.int16)
        self.buf0 = 0
        self.k = 0
        self.in_seg = False
        self.run = 0
        self.low = 0
        self.start_k = self.first_k = self.last_k = 0

    # ------------------------------------------------------------------ settings that change while running
    def set_limits(self, end_silence=None, cap=None):
        if end_silence is not None:
            self.end_silence = end_silence
        if cap is not None:
            self.cap = cap

    # ------------------------------------------------------------------ stream in
    def feed(self, pcm):
        out = []
        x = np.concatenate([self.tail, np.asarray(pcm, dtype=np.int16)])
        n = (len(x) // CH) * CH
        self.tail = x[n:]
        if n == 0:
            return out
        blk = x[:n]
        self.buf = np.concatenate([self.buf, blk])
        f = blk.astype(np.float32)
        if self.sos is not None:
            f, self.zi = sosfilt(self.sos, f, zi=self.zi)
            f = f.astype(np.float32)
        f = f / 32768.0
        # one chunk of look-back (context for the first chunk) and one dummy chunk at the end (the VAD wrapper zeroes the tail of the last chunk)
        audio = np.concatenate([self.prev, f, np.zeros(CH, np.float32)]).astype(np.float32)
        probs = vad_model()(audio).reshape(-1)[1:-1]
        self.prev = f[-CH:].copy()
        for p in probs:
            self._step(float(p), out)
        self._trim()
        return out

    def flush(self):
        out = []
        if self.in_seg:
            self._close(self.k - 1, "end", out)
        return out

    # ------------------------------------------------------------------ the rules
    def _step(self, p, out):
        k = self.k
        self.k += 1
        hot = p >= self.on
        if not self.in_seg:
            self.run = self.run + 1 if hot else 0
            if self.run >= self.start_run:
                self.first_k = k - self.start_run + 1
                self.start_k = max(self.buf0 // CH, self.first_k - self.pad_k)
                self.last_k = k
                self.in_seg = True
                self.low = 0
            return
        if hot:
            self.last_k, self.low = k, 0
        else:
            self.low += 1
        length = (k + 1 - self.start_k) * CH
        reason = None
        if (k - self.last_k) * CH >= self.end_silence * SR:
            reason = "silence"
        elif length >= self.cap * SR:
            reason = "cap"
        elif length >= (self.cap - self.soft_margin) * SR and self.low >= 4:
            reason = "soft_cap"  # near the cap: cut at the first short quiet moment instead of in the middle of a word
        if reason:
            self._close(k, reason, out)

    def _close(self, k, reason, out):
        end_k = min(k + 1, self.last_k + 1 + self.pad_k)
        speech = (self.last_k - self.first_k + 1) * CH
        self.in_seg, self.run = False, 0
        if speech < self.min_speech * SR:
            return
        a, b = self.start_k * CH - self.buf0, end_k * CH - self.buf0
        out.append({"pcm": self.buf[a:b].copy(), "t0": self.start_k * CH / SR, "t1": end_k * CH / SR,
                    "speech_t0": self.first_k * CH / SR, "speech_t1": (self.last_k + 1) * CH / SR,
                    "detected_t": (k + 1) * CH / SR, "wait_s": max(0.0, (k + 1 - (self.last_k + 1)) * CH / SR), "reason": reason})

    def _trim(self):
        keep_k = self.start_k if self.in_seg else self.k - self.pad_k - 6
        cut = max(0, keep_k * CH - self.buf0)
        if cut > 0:
            self.buf = self.buf[cut:]
            self.buf0 += cut


def cut_audio(pcm, **kw):
    """Cut a whole recording into speech segments (same rules as the live cutter)."""
    c = StreamCutter(**kw)
    segs = []
    step = SR * 10
    for i in range(0, len(pcm), step):
        segs += c.feed(pcm[i:i + step])
    segs += c.flush()
    return segs
