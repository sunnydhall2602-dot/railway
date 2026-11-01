# Railway Station Smart CCTV — Crowd & Stampede Early-Warning System

A proof-of-concept pipeline that watches a CCTV feed and predicts crowd-crush
/ stampede risk *before* it becomes critical, instead of only reacting once
it already has.

## Why this design

A raw head-count is a weak stampede signal — a packed concourse can be
perfectly orderly, while a small, fast-moving, chaotic crowd at a gate is
dangerous. So the pipeline looks at four independent signals and combines
them into one risk score:

| Signal | Module | What it catches |
|---|---|---|
| Local crowd density (grid, not global count) | `crowd_monitor/density.py` | Crush points — e.g. one gate jammed while the rest of the platform is empty |
| Optical-flow speed | `crowd_monitor/flow.py` | People running instead of walking |
| Optical-flow divergence | `crowd_monitor/flow.py` | Outward panic bursts (positive) and inward crushes (negative) |
| Optical-flow angular variance | `crowd_monitor/flow.py` | Chaotic/counter-flow motion vs. an orderly, coherent walking crowd |

`crowd_monitor/anomaly.py` turns those into a 0–100 composite score and a
`LOW / MEDIUM / HIGH / CRITICAL` level, **and** fits a short linear trend
over a rolling window to report a *predicted* level ~30s ahead — that's the
"minimal response time" part: control-room staff get a heads-up while the
crowd is still building up, not only once it's already a stampede.

`crowd_monitor/alert.py` only notifies on an escalation or a sustained
incident (not every frame), and never auto-triggers anything physical —
it's deliberately a decision-support signal for a human, because a false
stampede alarm can cause one.

## Pipeline

```
video source (file / webcam / RTSP)
        │
        ▼
 PersonDetector  (motion | hog | yolo backend)
        │  bounding boxes
        ▼
 DensityEstimator ──► per-cell crowd density + level
        │
 FlowAnalyzer     ──► speed / divergence / angular variance
        │
        ▼
 TemporalAnomalyScorer ──► current risk + predicted risk (trend-extrapolated)
        │
        ▼
 AlertManager ──► console log / webhook, de-duplicated by escalation
```

`crowd_monitor/pipeline.py` wires all of this together per frame and also
draws the annotated overlay (boxes, density heat cells, risk label).

## Detector backends

- **`motion`** (default): background subtraction + contours. No model
  download, runs on a Raspberry-Pi-class edge box. Used by the synthetic
  demo below.
- **`hog`**: OpenCV's built-in pedestrian detector. No download, better on
  real upright pedestrians than `motion`, slower.
- **`yolo`**: Ultralytics YOLOv8n filtered to the COCO "person" class.
  Best accuracy; needs `pip install ultralytics` and the model weights.

Swap backends with `--backend` on the CLI or `detector_backend=` on
`CrowdMonitorPipeline` — nothing else in the pipeline changes.

## Running it

```bash
pip install -r requirements.txt

# 1. Generate a synthetic test video (no real CCTV footage needed):
#    phase 1 = normal walking, phase 2 = crowd building up at a gate
#    (crush precursor), phase 3 = panic burst (the stampede event).
python sample_data/generate_synthetic_video.py

# 2. Run the pipeline on it and save an annotated copy:
python main.py --source sample_data/synthetic_stampede.mp4 \
    --output sample_data/synthetic_stampede_annotated.mp4

# 3. Or watch it live in a browser dashboard:
SOURCE=sample_data/synthetic_stampede.mp4 \
    uvicorn crowd_monitor.dashboard.app:app --port 8000
# then open http://localhost:8000/
```

On a live camera:

```bash
python main.py --source rtsp://<camera-ip>/stream1 \
    --backend yolo --webhook http://control-room.local/alerts
```

## What the synthetic demo shows

Running the pipeline on `synthetic_stampede.mp4` (see `tests/` for the
underlying unit tests) reproduces the intended escalation pattern:

- **Normal phase**: risk stays `LOW` (score ~15–20/100).
- **Buildup phase** (crowd converging on a gate): risk rises to
  `MEDIUM`/`HIGH` — a bottleneck/crush precursor, flagged *before* anyone
  panics.
- **Panic phase** (crowd bursts outward): risk hits `HIGH`, with the
  **predicted** level already reading `CRITICAL` several seconds ahead of
  the current frame — this lead time is the point of the predictive scorer.

## Tuning for a real deployment

Every threshold in `density.py` and `anomaly.py` is a dataclass field,
deliberately not hardcoded, because they depend on camera height/angle and
the real-world area each grid cell covers. Calibrate per-camera using a
short clip of that camera's normal foot traffic, then set the thresholds so
normal traffic scores comfortably under 25 (`LOW`).

## What this is *not*

This is a proof-of-concept pipeline, not a certified safety system. Before
any real deployment at a station:

- Validate on real CCTV footage for that specific camera placement, not
  just synthetic data.
- Keep a human in the loop — alerts are decision support, never an
  automated trigger for gates, announcements, or crowd-control actions.
- Add face/identity privacy controls and data-retention policy appropriate
  to applicable regulation before recording or storing footage.
- Add redundancy/monitoring for the pipeline itself (a crashed stampede
  detector is worse than no detector, because operators may rely on it).

## Tests

```bash
pytest tests/ -q
```
