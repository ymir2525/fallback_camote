"""Wake word "Hey Jarvis" on the AUDIO, using openWakeWord (the same stand-in model the old backend used). UNVERIFIED on the Pi.

    from camote_speech.wakeword import WakeWord
    ww = WakeWord()
    fired = ww.feed(chunk_int16_1280_samples)       # 80 ms chunks at 16 kHz; True once when the score passes 0.5, then 3 s quiet time

The old backend then waited for a short command window after the wake word (see docs/COMMANDS_AND_WAKE_WORD.md).
"""
import time

WAKEWORD_SCORE_THRESHOLD = 0.5
WAKEWORD_COOLDOWN_SECONDS = 3.0


class WakeWord:
    def __init__(self):
        from openwakeword.model import Model
        self.model = Model(wakeword_models=["hey_jarvis"], inference_framework="onnx")
        self.last = 0.0

    def feed(self, chunk_int16):
        scores = self.model.predict(chunk_int16)
        score = max(scores.values()) if scores else 0.0
        now = time.time()
        if score > WAKEWORD_SCORE_THRESHOLD and now - self.last > WAKEWORD_COOLDOWN_SECONDS:
            self.last = now
            return True
        return False
