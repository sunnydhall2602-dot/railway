"""Turns per-frame density + flow features into a stampede risk score.

Design choices, and why:

- Thresholds are fixed and documented (not learned from a "calibration
  period") so the system is useful from frame 1 — a baseline-only /
  pure-anomaly-detection design is blind during the exact window right
  after a camera comes online, which is an unacceptable gap for a safety
  system. Operators are expected to tune the thresholds per camera
  (see :mod:`crowd_monitor.density` and the weights below) to match that
  camera's height/angle and the real-world area it covers.
- A short rolling window of composite scores is fitted with a simple linear
  trend (least squares) and extrapolated forward, so the system can report
  a *predicted* risk level a few seconds ahead of the *current* one — this
  is the "minimal response time" requirement: control-room staff get a
  heads-up before the situation is already critical, not just a reactive
  alarm once it is.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .density import DensityResult
from .flow import FlowResult

LEVELS = ("LOW", "MEDIUM", "HIGH", "CRITICAL")

_DENSITY_LEVEL_SCORE = {"sparse": 0, "moderate": 5, "dense": 10, "critical": 15}


@dataclass
class RiskAssessment:
    score: float  # 0-100 composite anomaly score, current frame
    level: str  # current risk level
    predicted_score: float  # extrapolated score N seconds ahead
    predicted_level: str
    reasons: list[str] = field(default_factory=list)


@dataclass
class TemporalAnomalyScorer:
    fps: float = 15.0
    window_seconds: float = 8.0
    predict_ahead_seconds: float = 30.0

    # feature -> score normalization. Tune per deployment: these are
    # "pixels/frame of optical flow on a 320px-wide downscaled frame"
    # scale figures, not real-world speed units.
    speed_critical: float = 2.5
    divergence_critical: float = 0.05
    chaos_weight: float = 15.0

    def __post_init__(self) -> None:
        self._window: deque[tuple[float, float]] = deque(
            maxlen=max(int(self.window_seconds * self.fps), 1)
        )
        self._t: float = 0.0

    def update(self, density: DensityResult, flow: FlowResult) -> RiskAssessment:
        reasons: list[str] = []

        speed_score = min(flow.mean_speed / self.speed_critical, 1.0) * 40
        if speed_score > 20:
            reasons.append(f"high crowd movement speed ({flow.mean_speed:.2f})")

        divergence_score = min(abs(flow.divergence) / self.divergence_critical, 1.0) * 30
        if divergence_score > 15:
            direction = "outward burst (panic)" if flow.divergence > 0 else "inward crush"
            reasons.append(f"{direction} detected (divergence={flow.divergence:.3f})")

        chaos_score = flow.angular_variance * self.chaos_weight
        if chaos_score > self.chaos_weight * 0.6:
            reasons.append(f"incoherent/counter-flow motion (variance={flow.angular_variance:.2f})")

        density_score = _DENSITY_LEVEL_SCORE[density.level]
        if density.level in ("dense", "critical"):
            reasons.append(
                f"local crowd crush at grid cell {density.peak_cell} "
                f"({density.peak_cell_count} people)"
            )

        composite = speed_score + divergence_score + chaos_score + density_score
        composite = float(min(composite, 100.0))

        self._t += 1.0 / self.fps
        self._window.append((self._t, composite))

        level = self._classify(composite)
        predicted_score, predicted_level = self._predict()

        return RiskAssessment(
            score=composite,
            level=level,
            predicted_score=predicted_score,
            predicted_level=predicted_level,
            reasons=reasons,
        )

    def _classify(self, score: float) -> str:
        if score >= 75:
            return "CRITICAL"
        if score >= 50:
            return "HIGH"
        if score >= 25:
            return "MEDIUM"
        return "LOW"

    def _predict(self) -> tuple[float, str]:
        if len(self._window) < 3:
            current = self._window[-1][1] if self._window else 0.0
            return current, self._classify(current)

        ts = np.array([p[0] for p in self._window])
        scores = np.array([p[1] for p in self._window])
        slope, intercept = np.polyfit(ts, scores, 1)
        predicted = slope * (self._t + self.predict_ahead_seconds) + intercept
        predicted = float(np.clip(predicted, 0.0, 100.0))
        return predicted, self._classify(predicted)
