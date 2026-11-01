"""Minimal control-room dashboard: live annotated feed + alert log.

Run with:

    SOURCE=sample_data/synthetic_stampede.mp4 uvicorn crowd_monitor.dashboard.app:app --port 8000

Then open http://localhost:8000/ — the page shows the annotated MJPEG
stream and polls /alerts for the running alert log. This is a thin demo
UI, not meant to replace a real station control-room system; its job is
to prove the pipeline's output is usable in near-real-time.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import cv2
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from crowd_monitor.pipeline import CrowdMonitorPipeline

SOURCE = os.environ.get("SOURCE", "0")
BACKEND = os.environ.get("DETECTOR_BACKEND", "motion")
LOOP = os.environ.get("LOOP", "1") == "1"

app = FastAPI(title="Railway Crowd Monitor")

_state_lock = threading.Lock()
_latest_jpeg: bytes | None = None
_latest_status: dict = {"level": "LOW", "score": 0.0, "predicted_level": "LOW", "count": 0}
_alerts: list[dict] = []


def _video_loop() -> None:
    global _latest_jpeg, _latest_status

    source = int(SOURCE) if SOURCE.isdigit() else SOURCE
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    pipeline = CrowdMonitorPipeline(detector_backend=BACKEND, fps=fps)
    frame_delay = 1.0 / fps

    while True:
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
            _latest_status = {
                "level": result.risk.level,
                "score": round(result.risk.score, 1),
                "predicted_level": result.risk.predicted_level,
                "count": result.density.count,
                "reasons": result.risk.reasons,
            }
            if result.alert is not None:
                _alerts.append(result.alert.to_dict())
                del _alerts[:-50]

        time.sleep(frame_delay)

    cap.release()


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


@app.get("/", response_class=HTMLResponse)
def index():
    return (Path(__file__).parent / "templates" / "index.html").read_text()
