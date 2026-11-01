#!/usr/bin/env python3
"""Generates a synthetic top-down CCTV-style video with three phases:

1. normal: a handful of blobs ("people") walk coherently in one direction.
2. buildup: more blobs enter and bunch up near a choke point (a gate),
   motion slows — a classic pre-stampede crowd-crush precursor.
3. panic: the bunched-up blobs suddenly scatter outward in many directions
   at high speed — the stampede event itself.

This exists so the detection pipeline (crowd_monitor) can be exercised
end-to-end without needing real CCTV footage, which is not available in
this environment. The `motion` detector backend (background subtraction)
picks up these blobs the same way it would pick up real moving people on
a static camera.
"""

from __future__ import annotations

import random

import cv2
import numpy as np

WIDTH, HEIGHT = 480, 360
FPS = 15
PHASE_SECONDS = {"normal": 6, "buildup": 6, "panic": 6}
GATE = (WIDTH // 2, HEIGHT // 2)
BLOB_RADIUS = 7


def make_background(frame_idx: int) -> np.ndarray:
    frame = np.full((HEIGHT, WIDTH, 3), (40, 40, 40), dtype=np.uint8)
    cv2.rectangle(frame, (0, 0), (WIDTH, HEIGHT), (40, 40, 40), -1)
    # static "platform edge" line so the scene isn't perfectly featureless
    cv2.line(frame, (0, HEIGHT - 40), (WIDTH, HEIGHT - 40), (60, 60, 60), 2)
    cv2.rectangle(frame, (GATE[0] - 30, 0), (GATE[0] + 30, HEIGHT), (55, 55, 70), 1)
    return frame


class Person:
    def __init__(self, x: float, y: float):
        self.x, self.y = x, y
        self.vx, self.vy = 0.0, 0.0

    def step(self):
        self.x += self.vx
        self.y += self.vy
        self.x = min(max(self.x, BLOB_RADIUS), WIDTH - BLOB_RADIUS)
        self.y = min(max(self.y, BLOB_RADIUS), HEIGHT - BLOB_RADIUS)


def generate(path: str, seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)

    total_frames = sum(PHASE_SECONDS.values()) * FPS
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))

    people: list[Person] = [Person(random.uniform(20, WIDTH - 20), random.uniform(20, HEIGHT - 60)) for _ in range(8)]

    frame_idx = 0
    for phase, seconds in PHASE_SECONDS.items():
        for _ in range(seconds * FPS):
            frame = make_background(frame_idx)

            if phase == "normal":
                for p in people:
                    p.vx = 0.6 + random.uniform(-0.1, 0.1)
                    p.vy = random.uniform(-0.05, 0.05)
                    p.step()

            elif phase == "buildup":
                if frame_idx % 5 == 0 and len(people) < 40:
                    people.append(Person(random.uniform(0, WIDTH), HEIGHT - 10))
                for p in people:
                    dx, dy = GATE[0] - p.x, GATE[1] - p.y
                    dist = max((dx**2 + dy**2) ** 0.5, 1.0)
                    speed = 0.35 * min(dist / 80, 1.0)
                    p.vx, p.vy = speed * dx / dist, speed * dy / dist
                    p.step()

            else:  # panic
                for p in people:
                    dx, dy = p.x - GATE[0], p.y - GATE[1]
                    dist = max((dx**2 + dy**2) ** 0.5, 1.0)
                    jitter = random.uniform(-0.6, 0.6)
                    p.vx = 4.5 * dx / dist + jitter
                    p.vy = 4.5 * dy / dist + jitter
                    p.step()

            for p in people:
                cv2.circle(frame, (int(p.x), int(p.y)), BLOB_RADIUS, (230, 200, 180), -1)
                cv2.circle(frame, (int(p.x), int(p.y)), BLOB_RADIUS, (0, 0, 0), 1)

            cv2.putText(
                frame, f"phase={phase}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1
            )
            writer.write(frame)
            frame_idx += 1

    writer.release()
    print(f"Wrote {frame_idx} frames ({frame_idx / FPS:.1f}s) to {path}")


if __name__ == "__main__":
    generate("sample_data/synthetic_stampede.mp4")
