"""Wires detector + density + flow + anomaly + alert into one per-frame call."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from .alert import Alert, AlertManager
from .anomaly import RiskAssessment, TemporalAnomalyScorer
from .density import DensityEstimator, DensityResult
from .detector import Detection, PersonDetector
from .flow import FlowAnalyzer, FlowResult
from .hazard_detector import HazardDetection, HazardDetector
from .heatmap import DwellHeatmap
from .history import HistoryStore
from .social_distance import SocialDistanceChecker, SocialDistanceResult

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
    social_distance: SocialDistanceResult
    hazards: list[HazardDetection]
    hazard_alert: Optional[Alert]
    annotated_frame: np.ndarray


@dataclass
class CrowdMonitorPipeline:
    camera_id: str = "cam-01"
    detector_backend: str = "motion"
    fps: float = 15.0
    webhook_url: Optional[str] = None
    show_heatmap: bool = True
    show_social_distance: bool = True
    hazard_weights_path: Optional[str] = None
    history_db_path: Optional[str] = None
    history_record_every_seconds: float = 1.0

    def __post_init__(self) -> None:
        self.detector = PersonDetector(backend=self.detector_backend)
        self.density_estimator = DensityEstimator()
        self.flow_analyzer = FlowAnalyzer()
        self.anomaly_scorer = TemporalAnomalyScorer(fps=self.fps)
        self.heatmap = DwellHeatmap()
        self.social_distance_checker = SocialDistanceChecker()
        self.hazard_detector = HazardDetector(weights_path=self.hazard_weights_path)
        self.history = (
            HistoryStore(db_path=self.history_db_path, record_every_seconds=self.history_record_every_seconds)
            if self.history_db_path
            else None
        )

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
        social_distance = self.social_distance_checker.check(detections)
        self.heatmap.update(detections, frame.shape)
        hazards = self.hazard_detector.detect(frame)
        hazard_alert = self.alert_manager.process_hazard(hazards)

        if self.history is not None:
            now = time.time()
            self.history.record_frame(self.camera_id, density.count, risk, social_distance, ts=now)
            if alert is not None:
                self.history.record_alert(self.camera_id, alert)
            if hazard_alert is not None:
                self.history.record_alert(self.camera_id, hazard_alert)
            for hazard in hazards:
                self.history.record_event(self.camera_id, hazard.label, hazard.confidence, ts=now)

        annotated = self._annotate(frame, detections, density, risk, social_distance, hazards)

        return FrameResult(
            detections=detections,
            density=density,
            flow=flow,
            risk=risk,
            alert=alert,
            social_distance=social_distance,
            hazards=hazards,
            hazard_alert=hazard_alert,
            annotated_frame=annotated,
        )

    def _annotate(
        self,
        frame: np.ndarray,
        detections: list[Detection],
        density: DensityResult,
        risk: RiskAssessment,
        social_distance: SocialDistanceResult,
        hazards: list[HazardDetection],
    ) -> np.ndarray:
        out = frame.copy()
        color = _LEVEL_COLOR[risk.level]

        if self.show_heatmap:
            out = self.heatmap.overlay(out)

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

        for det in detections:
            cv2.rectangle(out, (det.x, det.y), (det.x + det.w, det.y + det.h), (255, 180, 0), 1)

        if self.show_social_distance:
            centroids = [det.centroid for det in detections]
            for i, j in social_distance.violations:
                p1 = tuple(int(v) for v in centroids[i])
                p2 = tuple(int(v) for v in centroids[j])
                cv2.line(out, p1, p2, (0, 0, 255), 1)
            for idx in {p for pair in social_distance.violations for p in pair}:
                cx, cy = map(int, centroids[idx])
                cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)

        for hazard in hazards:
            cv2.rectangle(out, (hazard.x, hazard.y), (hazard.x + hazard.w, hazard.y + hazard.h), (0, 69, 255), 2)
            cv2.putText(
                out, f"{hazard.label.upper()} {hazard.confidence:.2f}", (hazard.x, max(hazard.y - 6, 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 69, 255), 2,
            )

        cv2.rectangle(out, (0, 0), (w, 36), (30, 30, 30), -1)
        label = (
            f"n={density.count} risk={risk.level}({risk.score:.0f}) "
            f"pred={risk.predicted_level} sd={social_distance.violation_count}"
        )
        if hazards:
            label += f" HAZARD:{','.join(sorted({h.label for h in hazards}))}"
        cv2.putText(out, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return out
