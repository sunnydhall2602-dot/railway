"""Dense optical-flow based crowd motion analysis.

Density alone tells you a space is crowded; it does not tell you a stampede
is starting. The earliest signal of a stampede is usually in *motion*, not
headcount:

- a sudden spike in average speed (people running instead of walking)
- strong positive divergence (a point from which people are bursting
  outward — a panic epicenter)
- strong negative divergence (a point people are being crushed into — a
  bottleneck/crush risk)
- high angular variance (people moving in many different, conflicting
  directions — chaotic counter-flow instead of an orderly walking crowd)

This module only extracts these features per-frame; turning them into a
risk level over time is :mod:`crowd_monitor.anomaly`.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FlowResult:
    mean_speed: float = 0.0
    max_speed: float = 0.0
    divergence: float = 0.0
    angular_variance: float = 0.0  # 0 = everyone moving the same way, 1 = chaotic


class FlowAnalyzer:
    """Computes dense Farneback optical flow between consecutive frames."""

    def __init__(self, resize_width: int = 320, motion_threshold: float = 0.5):
        self.resize_width = resize_width
        self.motion_threshold = motion_threshold
        self._prev_gray: np.ndarray | None = None

    def reset(self) -> None:
        self._prev_gray = None

    def compute(self, frame: np.ndarray) -> FlowResult:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scale = min(1.0, self.resize_width / gray.shape[1])
        if scale < 1.0:
            gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale)

        if self._prev_gray is None or self._prev_gray.shape != gray.shape:
            self._prev_gray = gray
            return FlowResult()

        flow = cv2.calcOpticalFlowFarneback(
            self._prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        self._prev_gray = gray

        fx, fy = flow[..., 0], flow[..., 1]
        magnitude, angle = cv2.cartToPolar(fx, fy)
        motion_mask = magnitude > self.motion_threshold
        if not np.any(motion_mask):
            return FlowResult()

        mean_speed = float(magnitude[motion_mask].mean())
        max_speed = float(magnitude.max())

        dfx_dx = np.gradient(fx, axis=1)
        dfy_dy = np.gradient(fy, axis=0)
        divergence_field = dfx_dx + dfy_dy
        divergence = float(divergence_field[motion_mask].mean())

        angles = angle[motion_mask]
        resultant_length = float(
            np.hypot(np.sin(angles).mean(), np.cos(angles).mean())
        )
        angular_variance = 1.0 - resultant_length

        return FlowResult(
            mean_speed=mean_speed,
            max_speed=max_speed,
            divergence=divergence,
            angular_variance=angular_variance,
        )
