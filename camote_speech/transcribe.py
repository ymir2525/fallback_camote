"""Speech to text with the fine-tuned Whisper (faster-whisper / CTranslate2, int8, CPU). Same settings the tests used.

    from camote_speech.transcribe import Transcriber
    t = Transcriber("models/run13-ct2-int8")
    text = t.transcribe(pcm_int16_numpy_array_16khz)         # or t.transcribe_file("clip.wav")

Settings: language Tagalog ("tl", it handles Taglish), beam 1, temperature 0, domain prompt, no VAD inside (cut the audio before), pieces
with no_speech_prob >= 0.6 dropped. Measured on a PC CPU (4 threads) at about 2.7-3.1 s per short clip: NOT measured on the Pi.
"""
import math
import os
import wave

HERE = os.path.dirname(os.path.abspath(__file__))
PROMPT = open(os.path.join(HERE, "data", "domain_prompt.txt"), encoding="utf-8").read().strip()


class Transcriber:
    def __init__(self, model_dir, cpu_threads=4):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=cpu_threads)

    def transcribe(self, pcm_int16):
        import numpy as np
        x = np.asarray(pcm_int16).astype(np.float32) / 32768.0
        cap = min(int(math.ceil(len(x) / 16000 * 12)) + 10, 200)
        segs, _ = self.model.transcribe(x, language="tl", beam_size=1, vad_filter=False, condition_on_previous_text=False,
                                        initial_prompt=PROMPT, temperature=0.0, max_new_tokens=cap)
        return " ".join(s.text.strip() for s in segs if s.no_speech_prob < 0.6)

    def transcribe_file(self, path):
        import numpy as np
        with wave.open(path, "rb") as f:
            assert f.getframerate() == 16000 and f.getnchannels() == 1, "need 16 kHz mono wav"
            pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
        return self.transcribe(pcm)
