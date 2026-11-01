#!/usr/bin/env python3
"""CLI entrypoint: run the crowd/stampede monitor on a video file, webcam, or RTSP stream.

Examples
--------
Run on a recorded video, save annotated output, print alerts to the console::

    python main.py --source sample_data/synthetic_stampede.mp4 --output out.mp4

Run on a live RTSP camera feed and forward alerts to a control-room webhook::

    python main.py --source rtsp://192.168.1.50/stream1 --webhook http://ops.local/alerts
"""

from __future__ import annotations

import argparse
import logging
import sys

import cv2

from crowd_monitor.pipeline import CrowdMonitorPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        required=True,
        help="Video file path, RTSP/HTTP stream URL, or webcam index (e.g. 0).",
    )
    parser.add_argument("--camera-id", default="cam-01")
    parser.add_argument(
        "--backend",
        default="motion",
        choices=["motion", "hog", "yolo"],
        help="Detector backend: motion (default, no model download), hog, or yolo.",
    )
    parser.add_argument("--output", default=None, help="Optional path to save annotated video.")
    parser.add_argument("--webhook", default=None, help="Optional URL to POST alerts to.")
    parser.add_argument("--display", action="store_true", help="Show annotated frames in a window.")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (for testing).")
    parser.add_argument(
        "--hazard-weights", default=None,
        help="Path to a YOLO model trained on fire/smoke (or other hazard classes). "
        "Omit to run without hazard detection.",
    )
    parser.add_argument(
        "--history-db", default=None,
        help="Path to a SQLite file to log throttled frame stats, alerts, and hazard events. "
        "Omit to run without history logging.",
    )
    return parser.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logging.error("Could not open video source: %s", args.source)
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    pipeline = CrowdMonitorPipeline(
        camera_id=args.camera_id,
        detector_backend=args.backend,
        fps=fps,
        webhook_url=args.webhook,
        hazard_weights_path=args.hazard_weights,
        history_db_path=args.history_db,
    )

    writer = None
    frame_idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = pipeline.process_frame(frame)

            if args.output is not None:
                if writer is None:
                    h, w = result.annotated_frame.shape[:2]
                    writer = cv2.VideoWriter(
                        args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)
                    )
                writer.write(result.annotated_frame)

            if args.display:
                cv2.imshow("Crowd Monitor", result.annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1
            if args.max_frames and frame_idx >= args.max_frames:
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    logging.info("Processed %d frames.", frame_idx)
    return 0


if __name__ == "__main__":
    sys.exit(main())
