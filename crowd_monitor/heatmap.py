"""Dwell-time heatmap: where the crowd lingers, not just where it is right now.

Inspired by the "stationary point" heatmap in crowd-analysis tooling (e.g.
lewjiayi/Crowd-Analysis): a spot that is occupied frame after frame — a
ticket counter, a narrow gate, a queue corner — accumulates heat, while a
spot people only ever walk through stays cool even if it's briefly crowded.

This is a situational-awareness overlay for operators (chronic bottlenecks
worth redesigning), not part of the live stampede risk score in
:mod:`crowd_monitor.anomaly` — a spot can be legitimately "hot" (a busy but
safe ticket counter) without being at stampede risk.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .detector import Detection


@dataclass
class DwellHeatmap:
    decay: float = 0.98  # per-frame retention; ~115 frames (≈7.5s @15fps) to fall to half heat
    radius: int = 18  # splat radius in pixels around each detection centroid
    gain: float = 1.0  # heat added per frame a point stays occupied

    def __post_init__(self) -> None:
        self._heat: np.ndarray | None = None
        # theoretical steady-state heat for a point occupied every frame forever;
        # used to scale the overlay so "hot" means "persistently occupied", not
        # just "someone's here right now".
        self._steady_state = self.gain / max(1.0 - self.decay, 1e-6)

    def update(self, detections: list[Detection], frame_shape: tuple[int, int]) -> np.ndarray:
        height, width = frame_shape[:2]
        if self._heat is None or self._heat.shape != (height, width):
            self._heat = np.zeros((height, width), dtype=np.float32)

        self._heat *= self.decay

        if detections:
            occupancy = np.zeros((height, width), dtype=np.float32)
            for det in detections:
                cx, cy = det.centroid
                cv2.circle(occupancy, (int(cx), int(cy)), self.radius, 1.0, -1)
            occupancy = cv2.GaussianBlur(occupancy, (0, 0), sigmaX=self.radius / 2)
            self._heat += occupancy * self.gain

        return self._heat

    def overlay(self, frame: np.ndarray, alpha: float = 0.45, min_visible: float = 0.08) -> np.ndarray:
        """Blends the accumulated heat onto `frame`. `min_visible` (0-1 of
        steady-state) hides noise from people just passing through once."""
        if self._heat is None:
            return frame

        normalized = np.clip(self._heat / self._steady_state, 0.0, 1.0)
        visible = normalized > min_visible
        if not np.any(visible):
            return frame

        colored = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
        blended = cv2.addWeighted(frame, 1 - alpha, colored, alpha, 0)
        out = frame.copy()
        out[visible] = blended[visible]
        return out

    def hotspots(self, rows: int = 4, cols: int = 4, top_n: int = 3) -> list[tuple[int, int, float]]:
        """Returns up to `top_n` (row, col, normalized_heat) grid cells with the
        most persistent dwell — the chronic bottlenecks worth redesigning,
        as opposed to the live crush the density grid in `density.py` flags."""
        if self._heat is None:
            return []
        height, width = self._heat.shape
        cell_h, cell_w = height / rows, width / cols
        cells = []
        for r in range(rows):
            for c in range(cols):
                y0, y1 = int(r * cell_h), int((r + 1) * cell_h)
                x0, x1 = int(c * cell_w), int((c + 1) * cell_w)
                mean_heat = float(self._heat[y0:y1, x0:x1].mean()) / self._steady_state
                cells.append((r, c, mean_heat))
        cells.sort(key=lambda cell: cell[2], reverse=True)
        return cells[:top_n]
