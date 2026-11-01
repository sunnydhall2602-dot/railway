"""Grid-based crowd density estimation from a list of detections."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detector import Detection


@dataclass
class DensityResult:
    count: int
    grid: np.ndarray  # rows x cols occupancy counts
    peak_cell_count: int
    peak_cell: tuple[int, int]
    level: str  # "sparse" | "moderate" | "dense" | "critical"


@dataclass
class DensityEstimator:
    """Buckets detections into a coarse grid to find local crowd crush points.

    A single global head-count is a poor stampede signal: 200 people spread
    across a large concourse is fine, 20 people jammed into one gate is not.
    The grid lets the pipeline flag *where* density is building up, not just
    how many people are on camera overall.
    """

    rows: int = 4
    cols: int = 4
    # people-per-cell thresholds for classification; tune per-camera
    # (depends on camera height/angle and real-world area per cell).
    moderate_threshold: int = 2
    dense_threshold: int = 4
    critical_threshold: int = 7

    def compute(self, detections: list[Detection], frame_shape: tuple[int, int]) -> DensityResult:
        height, width = frame_shape[:2]
        grid = np.zeros((self.rows, self.cols), dtype=np.int32)

        cell_h = max(height / self.rows, 1e-6)
        cell_w = max(width / self.cols, 1e-6)

        for det in detections:
            cx, cy = det.centroid
            row = min(int(cy // cell_h), self.rows - 1)
            col = min(int(cx // cell_w), self.cols - 1)
            grid[row, col] += 1

        peak_idx = np.unravel_index(np.argmax(grid), grid.shape)
        peak_count = int(grid[peak_idx])

        if peak_count >= self.critical_threshold:
            level = "critical"
        elif peak_count >= self.dense_threshold:
            level = "dense"
        elif peak_count >= self.moderate_threshold:
            level = "moderate"
        else:
            level = "sparse"

        return DensityResult(
            count=len(detections),
            grid=grid,
            peak_cell_count=peak_count,
            peak_cell=(int(peak_idx[0]), int(peak_idx[1])),
            level=level,
        )
