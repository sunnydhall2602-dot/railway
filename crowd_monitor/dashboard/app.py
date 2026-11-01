"""Minimal control-room dashboard: live annotated feed + alert log.

Run with:

    SOURCE=sample_data/synthetic_stampede.mp4 uvicorn crowd_monitor.dashboard.app:app --port 8000

Then open http://localhost:8000/ — the page shows the annotated MJPEG
stream and polls /alerts for the running alert log. Use the upload form to
point the live pipeline at your own video file instead of the configured
SOURCE — it's saved locally (under UPLOAD_DIR) and processed by the same
crowd_monitor pipeline as a real camera feed would be.

This is a thin demo UI, not meant to replace a real station control-room
system; its job is to prove the pipeline's output is usable in
near-real-time.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from crowd_monitor.history import HistoryStore
from crowd_monitor.pipeline import CrowdMonitorPipeline

SOURCE = os.environ.get("SOURCE", "0")
BACKEND = os.environ.get("DETECTOR_BACKEND", "motion")
LOOP = os.environ.get("LOOP", "1") == "1"
HAZARD_WEIGHTS = os.environ.get("HAZARD_WEIGHTS") or None
HISTORY_DB = os.environ.get("HISTORY_DB") or None
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500MB

app = FastAPI(title="Railway Crowd Monitor")

_state_lock = threading.Lock()
_latest_jpeg: bytes | None = None
_latest_status: dict = {
    "level": "LOW", "score": 0.0, "predicted_level": "LOW", "count": 0,
    "sd_violations": 0, "hazards": [], "watching": SOURCE,
}
_alerts: list[dict] = []

_source_lock = threading.Lock()
_current_source: str = SOURCE
_source_changed = threading.Event()


def _set_source(new_source: str) -> None:
    with _source_lock:
        global _current_source
        _current_source = new_source
    _source_changed.set()


def _video_loop() -> None:
    global _latest_jpeg, _latest_status

    while True:
        with _source_lock:
            source_value = _current_source
        source = int(source_value) if str(source_value).isdigit() else source_value

        cap = cv2.VideoCapture(source)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        pipeline = CrowdMonitorPipeline(
            detector_backend=BACKEND, fps=fps,
            hazard_weights_path=HAZARD_WEIGHTS, history_db_path=HISTORY_DB,
        )
        frame_delay = 1.0 / fps
        _source_changed.clear()

        with _state_lock:
            _latest_status["watching"] = str(source_value)

        while not _source_changed.is_set():
            ok, frame = cap.read()
            if not ok:
                if LOOP and isinstance(source, str) and Path(source).exists():
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break

            result = pipeline.process_frame(frame)
            ok, buf = cv2.imencode(".jpg", result.annotated_frame)
            if not ok:
                continue

            with _state_lock:
                _latest_jpeg = buf.tobytes()
                _latest_status.update({
                    "level": result.risk.level,
                    "score": round(result.risk.score, 1),
                    "predicted_level": result.risk.predicted_level,
                    "count": result.density.count,
                    "reasons": result.risk.reasons,
                    "sd_violations": result.social_distance.violation_count,
                    "hazards": [h.label for h in result.hazards],
                })
                if result.alert is not None:
                    _alerts.append(result.alert.to_dict())
                    del _alerts[:-50]
                if result.hazard_alert is not None:
                    _alerts.append(result.hazard_alert.to_dict())
                    del _alerts[:-50]

            time.sleep(frame_delay)

        cap.release()
        if not _source_changed.is_set():
            # the clip ended and LOOP is off (or the source isn't a
            # re-seekable file) — idle until someone picks a new source
            # instead of spinning a tight empty loop.
            _source_changed.wait()


@app.on_event("startup")
def _start_background_loop() -> None:
    thread = threading.Thread(target=_video_loop, daemon=True)
    thread.start()


def _mjpeg_generator():
    while True:
        with _state_lock:
            frame = _latest_jpeg
        if frame is not None:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
        time.sleep(0.05)


@app.get("/stream")
def stream():
    return StreamingResponse(
        _mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/status")
def status():
    with _state_lock:
        return JSONResponse(_latest_status)


@app.get("/alerts")
def alerts():
    with _state_lock:
        return JSONResponse(list(reversed(_alerts)))


@app.get("/history")
def history(limit: int = 50):
    if not HISTORY_DB:
        return JSONResponse({"enabled": False, "alerts": [], "events": []})
    store = HistoryStore(db_path=HISTORY_DB)
    return JSONResponse({
        "enabled": True,
        "alerts": store.recent_alerts(limit=limit),
        "events": store.recent_events(limit=limit),
    })


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Saves the uploaded file locally and points the live pipeline at it.
    The file never leaves this machine — it's written under UPLOAD_DIR on
    the same server running this dashboard."""
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return JSONResponse(
            {"ok": False, "error": f"Unsupported file type {ext!r}. Use one of {sorted(ALLOWED_VIDEO_EXT)}."},
            status_code=400,
        )

    dest = UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}"
    size = 0
    try:
        with dest.open("wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    out.close()
                    dest.unlink(missing_ok=True)
                    return JSONResponse(
                        {"ok": False, "error": f"File too large (max {MAX_UPLOAD_BYTES // (1024*1024)}MB)."},
                        status_code=400,
                    )
                out.write(chunk)
    finally:
        await file.close()

    _set_source(str(dest))
    return JSONResponse({"ok": True, "filename": file.filename, "watching": str(dest)})


@app.post("/use-demo")
def use_demo():
    _set_source(SOURCE)
    return JSONResponse({"ok": True, "watching": SOURCE})


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "templates" / "index.html").read_text()
