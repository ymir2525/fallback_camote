"""Speaker recognition = "is this the enrolled enforcer?" (TitaNet-large through sherpa-onnx). Logic copied from hey-trafficam\\backend\\main.py
(same numbers); the roster / file storage / FastAPI parts are NOT copied: the Pi backend owns where voiceprints are saved.

Rules that matter (all measured by the owner on live two-person tests, kept as they were):
  - need at least MIN_SAMPLES (2 s at 16 kHz) of audio, otherwise no decision (returns None);
  - cosine similarity to the enrolled voiceprint; above the threshold = ENFORCER, below = DRIVER;
  - scores within STICKY_MARGIN of the threshold are ambiguous: keep the previous speaker, else UNKNOWN;
  - the threshold may drift at most +-0.05 around 0.55 during a session.
Voiceprints are PRIVATE: never commit them. Each Pi enrolls its own enforcers.
"""
import numpy as np

SPEAKER_MATCH_THRESHOLD = 0.55
MIN_SAMPLES = 32000
STICKY_MARGIN = 0.10
SESSION_THRESHOLD_MAX_DRIFT = 0.05
HIGH_CONFIDENCE_MARGIN = 0.20
ADAPTIVE_LEARNING_RATE = 0.05
OUTLIER_THRESHOLD = 0.5


class SpeakerGate:
    def __init__(self, model_path, num_threads=2):
        import sherpa_onnx
        cfg = sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=model_path, num_threads=num_threads, debug=False)
        self.ex = sherpa_onnx.SpeakerEmbeddingExtractor(cfg)

    def embedding(self, pcm_int16):
        s = self.ex.create_stream()
        s.accept_waveform(sample_rate=16000, waveform=np.asarray(pcm_int16).astype(np.float32) / 32768.0)
        s.input_finished()
        if not self.ex.is_ready(s):
            return None
        e = np.array(self.ex.compute(s))
        n = np.linalg.norm(e)
        return e / n if n > 0 else e  # unit length so every clip counts equally when averaging

    @staticmethod
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def enroll(self, clips):
        """clips: list of int16 arrays of the SAME person. Returns (voiceprint, used, dropped) or None. The caller saves the voiceprint (np.save)."""
        embs = [e for e in (self.embedding(c) for c in clips if len(c) >= 8000) if e is not None]
        n = len(embs)
        if n == 0:
            return None
        avg = [np.mean([self.cosine(embs[i], embs[j]) for j in range(n) if j != i]) if n > 1 else 1.0 for i in range(n)]
        keep = [e for e, a in zip(embs, avg) if a >= OUTLIER_THRESHOLD]
        if len(keep) < max(3, n // 2):
            return None  # samples disagree: ask the person to re-record
        return np.mean(keep, axis=0), len(keep), n - len(keep)

    def similarity(self, pcm_int16, voiceprint):
        if voiceprint is None or len(pcm_int16) < MIN_SAMPLES:
            return None
        e = self.embedding(pcm_int16)
        return None if e is None else self.cosine(e, voiceprint)


def session_threshold(enforcer_avg, driver_avg):
    if enforcer_avg is None or driver_avg is None:
        return SPEAKER_MATCH_THRESHOLD
    mid = (enforcer_avg + driver_avg) / 2
    return max(SPEAKER_MATCH_THRESHOLD - SESSION_THRESHOLD_MAX_DRIFT, min(SPEAKER_MATCH_THRESHOLD + SESSION_THRESHOLD_MAX_DRIFT, mid))


def classify(sim, previous_label="UNKNOWN", threshold=SPEAKER_MATCH_THRESHOLD):
    """-> "ENFORCER" | "DRIVER" | "UNKNOWN" (or previous_label when the score is in the ambiguous band or there is no score)."""
    if sim is None:
        return previous_label
    if abs(sim - threshold) < STICKY_MARGIN:
        return previous_label if previous_label != "UNKNOWN" else "UNKNOWN"
    return "ENFORCER" if sim > threshold else "DRIVER"
