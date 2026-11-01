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

Two more signals ride alongside the risk score as situational-awareness
overlays, not as inputs to it (a busy ticket counter can be legitimately
"hot" without being a stampede risk, so mixing them into the core score
would just add noise):

| Signal | Module | What it's for |
|---|---|---|
| Dwell heatmap | `crowd_monitor/heatmap.py` | Where the crowd *lingers* over time (a chronic bottleneck — a queue, a narrow gate) vs. where it only ever passes through. Decays over time, so a one-off crowd doesn't leave a permanent "hot" scar. |
| Social distancing | `crowd_monitor/social_distance.py` | Pairwise proximity violations, with a perspective-aware threshold (scales with detection height, since a fixed pixel distance means different real distances near vs. far from the camera). |

Both are inspired by the equivalent features in
[lewjiayi/Crowd-Analysis](https://github.com/lewjiayi/Crowd-Analysis), adapted
to this pipeline's own detector/pipeline shape instead of its YOLOv4+DeepSORT
stack.

Two more pieces round out "monitoring", not just "scoring":

| Piece | Module | What it's for |
|---|---|---|
| Hazard detection (fire/smoke) | `crowd_monitor/hazard_detector.py` | A second, independently-trained YOLO model that short-circuits straight to a CRITICAL alert — a fire doesn't wait for the crowd to start moving before it's an emergency. **Disabled by default** (no weights ship with this repo); point it at your own trained model with `--hazard-weights`. |
| History log | `crowd_monitor/history.py` | A SQLite log of throttled frame stats, every alert, and every hazard event — for shift handover, incident review, or justifying a chronic-bottleneck redesign. Opt in with `--history-db`; without it, nothing is written to disk. |

Both were suggested by the equivalent features in a reference CrowdGuard/
CrowdSafe project (optional smoke/fight model hooks, SQLite logging) —
adapted the same way: as opt-in additions that default to off/no-op rather
than silently depending on a model this repo doesn't ship.

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

`crowd_monitor/pipeline.py` wires all of this together per frame — plus the
optional hazard detector and history log below — and also draws the
annotated overlay (dwell heatmap, density heat cells, boxes, social-distance
violation lines, hazard boxes, risk label). Set `show_heatmap=False` /
`show_social_distance=False` on `CrowdMonitorPipeline` to turn either
overlay off without disabling the underlying computation.

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
pip install -r requirements-yolo.txt  # only needed for --backend yolo
python main.py --source rtsp://<camera-ip>/stream1 \
    --backend yolo --webhook http://control-room.local/alerts
```

With history logging and hazard detection turned on (both opt-in):

```bash
python main.py --source rtsp://<camera-ip>/stream1 \
    --history-db cam01_history.db \
    --hazard-weights /path/to/your/fire-smoke-yolo.pt
```

Or via the dashboard:

```bash
SOURCE=rtsp://<camera-ip>/stream1 \
    HISTORY_DB=cam01_history.db \
    HAZARD_WEIGHTS=/path/to/your/fire-smoke-yolo.pt \
    uvicorn crowd_monitor.dashboard.app:app --port 8000
# /history returns the logged alerts + hazard events; the page also
# shows a hazard banner if HAZARD_WEIGHTS detects anything.
```

## Deploying the dashboard as a real website

`Procfile` and `railway.json` are set up for a one-click deploy on
[Railway](https://railway.app):

1. On railway.app, **New Project → Deploy from GitHub repo**, pick this repo.
   Railway's Nixpacks builder auto-detects Python, installs
   `requirements.txt`, and runs the `Procfile`'s `web:` command. No
   Dockerfile needed.
2. Under the service's **Variables**, set at minimum:
   - `SOURCE` — defaults to `0` (webcam), which doesn't exist on a cloud
     server. Set it to `sample_data/synthetic_stampede.mp4` to get a working
     demo out of the box, or to your camera's URL (see the caveat below).
   - `DETECTOR_BACKEND` — `motion` (default, no extra install) or `hog`.
     `yolo` also needs `requirements-yolo.txt` installed, which Nixpacks
     won't do unless you add it to the build — pip-install both files in a
     custom build command if you need it.
   - `LOOP=1` to keep replaying a video file forever (the default).
   - `HISTORY_DB` / `HAZARD_WEIGHTS` if you want those features — see
     the caveat on persistence below.
3. Railway assigns a public URL automatically; open it and you'll see the
   same dashboard described above (`/`, `/stream`, `/status`, `/alerts`,
   `/history`).

This works the same way on Render, Fly.io, or any host that runs a Procfile
or a plain `uvicorn` command — Railway isn't required, it's just what's
set up here (and matches this repo's name).

**Two things that don't "just work" on a cloud PaaS, by the nature of the
problem, not a bug in this code:**

- **Reaching a real camera.** An RTSP feed from a CCTV camera at a station
  is normally on a private LAN, not exposed to the public internet (for
  good reason). A cloud-hosted dashboard can't reach `rtsp://192.168.x.x/...`
  — that address means something different from Railway's network. For a
  real deployment, either run this dashboard on an edge box *on the same
  network as the camera* (the architecture covered earlier), or set up a
  secure tunnel/VPN from that network to the cloud service. Don't expose
  the camera's RTSP port to the public internet to work around this.
- **`HISTORY_DB`'s SQLite file.** Most PaaS containers have ephemeral local
  disk — it's wiped on every redeploy/restart. Attach a persistent volume
  (Railway has a Volumes feature) and point `HISTORY_DB` at a path inside
  it, or the history resets constantly.

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
