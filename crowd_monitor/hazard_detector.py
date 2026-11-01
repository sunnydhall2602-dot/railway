"""Optional secondary detector for fire/smoke (or other custom hazard classes).

Crowd risk isn't the only thing that turns a station into a stampede — a
platform fire or visible smoke is a more direct, more urgent trigger, and
one the crowd-motion signals in `flow.py`/`anomaly.py` won't reliably catch
(people may not even be reacting yet). This hook runs a second,
independently-trained YOLO model alongside the crowd pipeline and, when it
fires, short-circuits straight to a CRITICAL alert via
`alert.AlertManager.process_hazard` — bypassing the crowd risk score
entirely, because a confirmed fire is critical regardless of how calm the
crowd still looks.

No hazard-detection weights ship with this repo: training a reliable
fire/smoke detector needs its own labeled dataset (there are public
fire/smoke YOLO datasets to start from) and real validation, which is out
of scope here. Point `weights_path` at your own trained model to activate
this; with `weights_path=None` (the default) it is a strict no-op —
`enabled` is `False` and `detect()` always returns `[]` — identical to not
having this feature at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_HAZARD_CLASSES = {"fire", "smoke"}


@dataclass
class HazardDetection:
    label: str
    confidence: float
    x: int
    y: int
    w: int
    h: int


@dataclass
class HazardDetector:
    weights_path: Optional[str] = None
    confidence_threshold: float = 0.5
    # Restrict to these class names from the model's own label map; None
    # accepts whatever classes the model was trained on.
    classes: Optional[set] = None

    def __post_init__(self) -> None:
        self._model = None
        if self.weights_path:
            try:
                from ultralytics import YOLO

                self._model = YOLO(self.weights_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Could not load hazard model from {self.weights_path!r}: {exc}. "
                    "Install `ultralytics` and confirm the weights path, or leave "
                    "weights_path=None to run without hazard detection."
                ) from exc

    @property
    def enabled(self) -> bool:
        return self._model is not None

    def detect(self, frame: np.ndarray) -> list[HazardDetection]:
        if self._model is None:
            return []

        results = self._model.predict(frame, verbose=False, conf=self.confidence_threshold)
        detections = []
        for r in results:
            names = r.names
            for box in r.boxes:
                label = names[int(box.cls[0])]
                if self.classes and label not in self.classes:
                    continue
                confidence = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()[:4]
                detections.append(
                    HazardDetection(
                        label=label, confidence=confidence,
                        x=int(x1), y=int(y1), w=int(x2 - x1), h=int(y2 - y1),
                    )
                )
        return detections
