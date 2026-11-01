"""Wires detector + density + flow + anomaly + alert into one per-frame call."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .alert import Alert, AlertManager
from .anomaly import RiskAssessment, TemporalAnomalyScorer
from .density import DensityEstimator, DensityResult
from .detector import Detection, PersonDetector
from .flow import FlowAnalyzer, FlowResult

_LEVEL_COLOR = {
    "LOW": (0, 200, 0),
    "MEDIUM": (0, 200, 255),
    "HIGH": (0, 100, 255),
    "CRITICAL": (0, 0, 255),
}


@dataclass
class FrameResult:
    detections: list[Detection]
    density: DensityResult
    flow: FlowResult
    risk: RiskAssessment
    alert: Optional[Alert]
    annotated_frame: np.ndarray


@dataclass
class CrowdMonitorPipeline:
    camera_id: str = "cam-01"
    detector_backend: str = "motion"
    fps: float = 15.0
    webhook_url: Optional[str] = None

    def __post_init__(self) -> None:
        self.detector = PersonDetector(backend=self.detector_backend)
        self.density_estimator = DensityEstimator()
        self.flow_analyzer = FlowAnalyzer()
        self.anomaly_scorer = TemporalAnomalyScorer(fps=self.fps)

        webhook = None
        if self.webhook_url:
            from .alert import http_webhook

            webhook = http_webhook(self.webhook_url)
        self.alert_manager = AlertManager(camera_id=self.camera_id, webhook=webhook)

    def process_frame(self, frame: np.ndarray) -> FrameResult:
        detections = self.detector.detect(frame)
        density = self.density_estimator.compute(detections, frame.shape)
        flow = self.flow_analyzer.compute(frame)
        risk = self.anomaly_scorer.update(density, flow)
        alert = self.alert_manager.process(risk)

        annotated = self._annotate(frame, detections, density, risk)

        return FrameResult(
            detections=detections,
            density=density,
            flow=flow,
            risk=risk,
            alert=alert,
            annotated_frame=annotated,
        )

    def _annotate(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        density: DensityResult,
        risk: RiskAssessment,
    ) -> np.ndarray:
        out = frame.copy()
        color = _LEVEL_COLOR[risk.level]

        for det in detections:
            cv2.rectangle(out, (det.x, det.y), (det.x + det.w, det.y + det.h), (255, 180, 0), 1)

        h, w = out.shape[:2]
        cell_h, cell_w = h / density.grid.shape[0], w / density.grid.shape[1]
        for r in range(density.grid.shape[0]):
            for c in range(density.grid.shape[1]):
                if density.grid[r, c] >= self.density_estimator.moderate_threshold:
                    x0, y0 = int(c * cell_w), int(r * cell_h)
                    x1, y1 = int((c + 1) * cell_w), int((r + 1) * cell_h)
                    overlay = out.copy()
                    cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 255), -1)
                    out = cv2.addWeighted(overlay, 0.15, out, 0.85, 0)

        cv2.rectangle(out, (0, 0), (w, 36), (30, 30, 30), -1)
        label = (
            f"count={density.count} risk={risk.level} score={risk.score:.0f} "
            f"predicted({self.anomaly_scorer.predict_ahead_seconds:.0f}s)={risk.predicted_level}"
        )
        cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        return out
