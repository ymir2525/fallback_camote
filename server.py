"""FALLBACK_CAMOTE server: serves the mobile-style page, takes the microphone stream, runs the live path, and runs the after-stop pass on the uploaded video.

Run:   run.ps1      (or:  <python> server.py)      then open  http://127.0.0.1:8765/
Separate from hey-trafficam and from the CAMOTE fine-tuning/gathering folders: it only READS the Whisper model folder.
"""
import asyncio
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import uvicorn  # noqa: E402
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402

import engine as E  # noqa: E402

ENGINE = None
LOOP = None
QUEUES = set()            # one asyncio.Queue per connected page


def broadcast(ev):
    """Thread-safe: after-stop progress goes to every open page."""
    if LOOP is None:
        return
    for q in list(QUEUES):
        LOOP.call_soon_threadsafe(q.put_nowait, ev)


@asynccontextmanager
async def lifespan(_app):
    global ENGINE, LOOP
    LOOP = asyncio.get_running_loop()
    ENGINE = await LOOP.run_in_executor(None, E.Engine)       # loads Whisper before the first page request is served
    print(f"[fallback] ready: http://{os.environ.get('FALLBACK_HOST', '127.0.0.1')}:{os.environ.get('FALLBACK_PORT', '8765')}/", flush=True)
    yield


app = FastAPI(title="FALLBACK_CAMOTE", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")


@app.middleware("http")
async def no_cache(request, call_next):                      # while testing: always load the newest page code
    resp = await call_next(request)
    if request.url.path.startswith(("/static", "/samples")) or request.url.path == "/":
        resp.headers["Cache-Control"] = "no-store"
    return resp


(HERE / "samples").mkdir(exist_ok=True)
app.mount("/samples", StaticFiles(directory=str(HERE / "samples")), name="samples")      # the "Play sample conversation" recording


@app.get("/", response_class=HTMLResponse)
def index():
    return (HERE / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"ok": ENGINE is not None, "whisper": os.path.basename(E.CFG["WHISPER_DIR"].rstrip("/\\")), "wake_word": bool(ENGINE and ENGINE.wake_available)}


@app.get("/api/offenses")
def offenses():
    return E.offense_list()


@app.get("/api/sessions")
def sessions():
    return E.list_sessions()


def _case(sid):
    p = E.SESSIONS / sid / "case.json"
    if not p.exists() or "/" in sid or "\\" in sid or ".." in sid:
        raise HTTPException(404, "no such case")
    return p


@app.get("/api/session/{sid}")
def session(sid: str):
    meta = json.loads(_case(sid).read_text(encoding="utf-8"))
    d = E.SESSIONS / sid
    meta["video"] = E.find_video(d) is not None
    meta["finals"] = {f.stem.replace("final_", ""): json.loads(f.read_text(encoding="utf-8")) for f in d.glob("final_*.json")}
    meta.pop("live_events", None)
    return meta


@app.get("/api/session/{sid}/video")
def session_video(sid: str):
    _case(sid)
    v = E.find_video(E.SESSIONS / sid)
    if v is None:
        raise HTTPException(404, "no video")
    return FileResponse(str(v), media_type="video/mp4" if v.suffix == ".mp4" else "video/webm")


def _start_final(sid, source):
    def go():
        try:
            broadcast({"type": "final_start", "sid": sid, "source": source})
            rep = E.run_final(ENGINE, sid, source,
                              progress=lambda i, n: broadcast({"type": "final_progress", "sid": sid, "source": source, "done": i, "total": n}))
            # no line-by-line output: the whole transcript is sent once, when everything is transcribed
            broadcast({"type": "final_done", "sid": sid, "source": source, "report": rep})
        except Exception as ex:
            broadcast({"type": "final_error", "sid": sid, "source": source, "message": repr(ex)})
    threading.Thread(target=go, daemon=True).start()


@app.post("/api/session/{sid}/video")
async def upload_video(sid: str, video: UploadFile = File(...), vid: str = Form("")):
    p = _case(sid)
    d = p.parent
    ext = ".mp4" if (video.content_type or "").startswith("video/mp4") else ".webm"
    raw = d / f"video_raw{ext}.tmp"
    with open(raw, "wb") as f:
        while chunk := await video.read(1 << 20):
            f.write(chunk)
    # Chrome's recorder writes a WebM without a length, so players show a wrong duration; copy it into a proper file (no re-encode)
    if not E.remux_with_duration(raw, d / f"video{ext}"):
        os.replace(raw, d / f"video{ext}")
    raw.unlink(missing_ok=True)
    meta = json.loads(p.read_text(encoding="utf-8"))
    meta["video_vid"] = vid or None
    p.write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")
    return {"ok": True, "sid": sid}          # stored only: the transcript starts when the ticket is saved


@app.post("/api/session/{sid}/save")
async def save_ticket(sid: str, req: Request):
    """Ticket saved: keep the ticket form next to the case, then make the after-stop transcript (video if there is one)."""
    p = _case(sid)
    try:
        ticket = await req.json()
    except Exception:
        ticket = {}
    (p.parent / "ticket.json").write_text(json.dumps(ticket, indent=1, ensure_ascii=False), encoding="utf-8")
    source = "video" if E.find_video(p.parent) is not None else "stream"
    _start_final(sid, source)
    return {"ok": True, "source": source}


@app.post("/api/session/{sid}/final")
def final_from_stream(sid: str, source: str = "stream"):
    _case(sid)
    if source not in ("stream", "video"):
        raise HTTPException(400, "source must be stream or video")
    _start_final(sid, source)
    return {"ok": True}


@app.websocket("/ws")
async def ws(sock: WebSocket):
    await sock.accept()
    q = asyncio.Queue()
    QUEUES.add(q)
    live = E.Live(ENGINE, lambda ev: LOOP.call_soon_threadsafe(q.put_nowait, ev))

    async def sender():
        while True:
            await sock.send_json(await q.get())

    task = asyncio.create_task(sender())
    c = E.CFG
    await sock.send_json({"type": "hello", "whisper": os.path.basename(c["WHISPER_DIR"].rstrip("/\\")), "wake_word": ENGINE.wake_available,
                          "commands": E.COMMANDS, "config": {k: c[k] for k in ("ON", "END_SILENCE", "CAP", "AWAKE_SECONDS", "AWAKE_END_SILENCE", "AWAKE_CAP",
                                                                                 "WAKE_MODE", "WAKE_THRESHOLD", "JOIN_LINES", "LIVE_BEAM", "FINAL_BEAM", "MERGE_GAP", "MAX_WAIT")}})
    try:
        while True:
            m = await sock.receive()
            if m.get("type") == "websocket.disconnect":
                break
            if m.get("bytes") is not None:
                live.feed(m["bytes"])
            elif m.get("text"):
                j = json.loads(m["text"])
                t = j.get("type")
                if t == "marker":
                    live.mark(str(j.get("vid", "")))
                elif t == "command":
                    live.do_command(str(j.get("name", "")), "app")
                elif t == "settings":
                    live.settings(**{k: v for k, v in j.items() if k != "type"})
                elif t == "reset":
                    live.recording = False
                    live.history, live.live_labels, live.case = [], [], None
                    live.awake_until = -1.0
    except WebSocketDisconnect:
        pass
    finally:
        task.cancel()
        QUEUES.discard(q)
        live.close()


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("FALLBACK_HOST", "127.0.0.1"), port=int(os.environ.get("FALLBACK_PORT", "8765")), log_level="warning")
