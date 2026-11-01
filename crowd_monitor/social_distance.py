"""Pairwise social-distance violation checking between detections.

Distance is measured between detection centroids in pixels. A single fixed
pixel threshold is wrong across a frame with any perspective (someone near
the camera is "bigger" per real-world metre than someone far away), so by
default the threshold scales with each frame's median detection height as a
crude proxy for scale — a taller box means the camera sees that depth
larger, so a given real-world distance covers more pixels there. This is
the same compromise `lewjiayi/Crowd-Analysis` makes with its
``HIGH_CAM``/center-vs-edge distance modes; it is not a substitute for a
calibrated homography if the camera has strong perspective.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from itertools import combinations

from .detector import Detection


@dataclass
class SocialDistanceResult:
    violations: list[tuple[int, int]] = field(default_factory=list)  # index pairs into detections
    violation_count: int = 0
    people_in_violation: int = 0
    violation_ratio: float = 0.0  # people_in_violation / total people, 0 if fewer than 2 people
    threshold_px: float = 0.0


@dataclass
class SocialDistanceChecker:
    min_distance_px: float = 60.0
    use_dynamic_scale: bool = True
    scale_factor: float = 1.2  # min_distance = max(min_distance_px, scale_factor * median detection height)

    def check(self, detections: list[Detection]) -> SocialDistanceResult:
        if len(detections) < 2:
            return SocialDistanceResult(threshold_px=self.min_distance_px)

        threshold = self.min_distance_px
        if self.use_dynamic_scale:
            median_height = statistics.median(det.h for det in detections)
            threshold = max(threshold, self.scale_factor * median_height)

        centroids = [det.centroid for det in detections]
        violations: list[tuple[int, int]] = []
        violators: set[int] = set()

        for i, j in combinations(range(len(detections)), 2):
            dx = centroids[i][0] - centroids[j][0]
            dy = centroids[i][1] - centroids[j][1]
            if (dx * dx + dy * dy) ** 0.5 < threshold:
                violations.append((i, j))
                violators.add(i)
                violators.add(j)

        return SocialDistanceResult(
            violations=violations,
            violation_count=len(violations),
            people_in_violation=len(violators),
            violation_ratio=len(violators) / len(detections),
            threshold_px=threshold,
        )
