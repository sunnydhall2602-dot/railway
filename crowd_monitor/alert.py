"""Alert dispatch with escalation-aware de-duplication.

A naive "alert every frame the risk is HIGH" design would fire ~15
notifications a second and get muted/ignored by a control room within
minutes. This manager only alerts on:

- a rise in risk level (LOW -> MEDIUM -> HIGH -> CRITICAL), immediately, or
- the level being sustained at MEDIUM+ for longer than `repeat_seconds`
  (so a genuinely ongoing incident still gets periodic re-notified).

It intentionally never auto-escalates past notifying a human: this is a
decision-support signal for the control room, not an automated trigger for
anything physical (gates, announcements, etc.), because a false positive
stampede alarm can itself cause a stampede.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from .anomaly import RiskAssessment
from .hazard_detector import HazardDetection

logger = logging.getLogger("crowd_monitor.alert")

_LEVEL_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2, "CRITICAL": 3}


@dataclass
class Alert:
    camera_id: str
    level: str
    score: float
    predicted_level: str
    reasons: list[str]
    timestamp: float

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "level": self.level,
            "score": round(self.score, 1),
            "predicted_level": self.predicted_level,
            "reasons": self.reasons,
            "timestamp": self.timestamp,
        }


@dataclass
class AlertManager:
    camera_id: str = "cam-01"
    repeat_seconds: float = 20.0
    hazard_repeat_seconds: float = 10.0
    webhook: Optional[Callable[[Alert], None]] = None

    _last_level: str = field(default="LOW", init=False)
    _last_notified_at: float = field(default=0.0, init=False)
    _hazard_last_notified_at: float = field(default=0.0, init=False)
    history: list[Alert] = field(default_factory=list, init=False)

    def process(self, assessment: RiskAssessment) -> Optional[Alert]:
        now = time.time()
        escalated = _LEVEL_RANK[assessment.level] > _LEVEL_RANK[self._last_level]
        sustained = (
            assessment.level in ("MEDIUM", "HIGH", "CRITICAL")
            and (now - self._last_notified_at) >= self.repeat_seconds
        )

        self._last_level = assessment.level

        if not (escalated or sustained):
            return None

        alert = Alert(
            camera_id=self.camera_id,
            level=assessment.level,
            score=assessment.score,
            predicted_level=assessment.predicted_level,
            reasons=assessment.reasons,
            timestamp=now,
        )
        self._last_notified_at = now
        self.history.append(alert)
        self._dispatch(alert)
        return alert

    def process_hazard(self, hazards: list[HazardDetection]) -> Optional[Alert]:
        """Fire/smoke detections always escalate straight to CRITICAL,
        independent of the crowd risk score — a confirmed fire doesn't wait
        for the crowd to start moving. Re-notifies every `hazard_repeat_seconds`
        while the hazard is still in frame, and resets as soon as it clears."""
        if not hazards:
            self._hazard_last_notified_at = 0.0
            return None

        now = time.time()
        if (now - self._hazard_last_notified_at) < self.hazard_repeat_seconds:
            return None
        self._hazard_last_notified_at = now

        labels = sorted({h.label for h in hazards})
        top_confidence = max(h.confidence for h in hazards)
        alert = Alert(
            camera_id=self.camera_id,
            level="CRITICAL",
            score=100.0,
            predicted_level="CRITICAL",
            reasons=[f"{label} detected (confidence={top_confidence:.2f})" for label in labels],
            timestamp=now,
        )
        self.history.append(alert)
        self._dispatch(alert)
        return alert

    def _dispatch(self, alert: Alert) -> None:
        log_fn = logger.warning if alert.level in ("HIGH", "CRITICAL") else logger.info
        log_fn("STAMPEDE_RISK %s", json.dumps(alert.to_dict()))
        if self.webhook is not None:
            try:
                self.webhook(alert)
            except Exception:  # pragma: no cover - never let a bad webhook kill the pipeline
                logger.exception("Alert webhook failed")


def http_webhook(url: str) -> Callable[[Alert], None]:
    """Convenience factory: POST the alert JSON to a control-room endpoint."""
    import requests

    def _send(alert: Alert) -> None:
        requests.post(url, json=alert.to_dict(), timeout=3)

    return _send
