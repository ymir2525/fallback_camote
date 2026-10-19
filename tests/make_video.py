"""Builds a small test video (VP8 video + Opus audio in WebM: the same container and codecs Chrome's MediaRecorder produces) from 16 kHz int16 audio."""
import fractions

import av
import numpy as np


def make_webm(pcm16k, path, fps=5, size=(320, 180)):
    out = av.open(str(path), "w", format="webm")
    vs = out.add_stream("libvpx", rate=fps)
    vs.width, vs.height, vs.pix_fmt = size[0], size[1], "yuv420p"
    aus = out.add_stream("libopus", rate=48000)
    aus.layout = "mono"
    rs = av.AudioResampler(format="s16", layout="mono", rate=48000, frame_size=960)
    secs = int(np.ceil(len(pcm16k) / 16000))
    vframe_i = 0
    apos = 0
    for s in range(secs):
        for _ in range(fps):                                 # one second of grey video frames, one per 1/fps
            img = np.full((size[1], size[0], 3), 40 + (vframe_i % 50), dtype=np.uint8)
            vf = av.VideoFrame.from_ndarray(img, format="rgb24")
            vf.pts = vframe_i
            vf.time_base = fractions.Fraction(1, fps)
            for pkt in vs.encode(vf):
                out.mux(pkt)
            vframe_i += 1
        chunk = pcm16k[s * 16000:(s + 1) * 16000]
        if len(chunk) == 0:
            continue
        af = av.AudioFrame.from_ndarray(chunk.reshape(1, -1).astype(np.int16), format="s16", layout="mono")
        af.sample_rate = 16000
        af.pts = apos
        apos += len(chunk)
        af.time_base = fractions.Fraction(1, 16000)
        for r in rs.resample(af):
            for pkt in aus.encode(r):
                out.mux(pkt)
    for r in rs.resample(None) or []:
        for pkt in aus.encode(r):
            out.mux(pkt)
    for pkt in vs.encode(None):
        out.mux(pkt)
    for pkt in aus.encode(None):
        out.mux(pkt)
    out.close()
