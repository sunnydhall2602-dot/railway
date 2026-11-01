"""Person/crowd-blob detection backends.

Three interchangeable backends so the pipeline can run on anything from a
Raspberry-Pi-class edge box (motion backend, no model download) up to a GPU
box running a real detector (yolo backend):

- ``motion``: background subtraction + contours. No model, no download,
  works on any scene with a mostly-static camera. Used as the default/
  fallback and for the synthetic test video.
- ``hog``: OpenCV's built-in HOG + SVM pedestrian detector. No download,
  reasonably accurate on upright pedestrians, slower than motion.
- ``yolo``: Ultralytics YOLOv8n, filtered to the COCO "person" class.
  Best accuracy, needs the model weights and a bit more compute.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detection:
    x: int
    y: int
    w: int
    h: int

    @property
    def centroid(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def area(self) -> int:
        return self.w * self.h


class PersonDetector:
    """Detects people/crowd blobs in a frame using a pluggable backend."""

    def __init__(self, backend: str = "motion", min_area: int = 150):
        self.backend = backend
        self.min_area = min_area
        self._bg_subtractor = None
        self._hog = None
        self._yolo = None

        if backend == "motion":
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=16, detectShadows=False
            )
        elif backend == "hog":
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        elif backend == "yolo":
            try:
                from ultralytics import YOLO

                self._yolo = YOLO("yolov8n.pt")
            except Exception as exc:  # pragma: no cover - exercised only without deps
                raise RuntimeError(
                    "YOLO backend requested but ultralytics/model weights are "
                    "unavailable. Install `ultralytics` and ensure network access "
                    "to download yolov8n.pt, or use backend='hog'/'motion'."
                ) from exc
        else:
            raise ValueError(f"Unknown detector backend: {backend!r}")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.backend == "motion":
            return self._detect_motion(frame)
        if self.backend == "hog":
            return self._detect_hog(frame)
        return self._detect_yolo(frame)

    def _detect_motion(self, frame: np.ndarray) -> list[Detection]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        mask = self._bg_subtractor.apply(gray)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            detections.append(Detection(x, y, w, h))
        return detections

    def _detect_hog(self, frame: np.ndarray) -> list[Detection]:
        rects, _ = self._hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        return [Detection(int(x), int(y), int(w), int(h)) for (x, y, w, h) in rects]

    def _detect_yolo(self, frame: np.ndarray) -> list[Detection]:
        results = self._yolo.predict(frame, classes=[0], verbose=False)
        detections = []
        for r in results:
            for box in r.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = box[:4]
                detections.append(Detection(int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        return detections
