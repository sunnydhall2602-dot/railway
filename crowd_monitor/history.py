"""SQLite history store: the live pipeline is a snapshot, this is the log.

Everything else in this package answers "what is the risk right now?" A
control room also needs "how often did this happen, and when?" — for
shift handover, incident review, or justifying a redesign of a chronic
bottleneck. This stores a throttled trail of frames, every alert, and any
hazard event (see `hazard_detector.py`) to a local SQLite file.

Frame rows are deliberately NOT written every frame — at 15fps that's
~1.3M rows/day per camera for no benefit. `record_every_seconds` thins
that to one row per interval; alerts and events are always recorded in
full since they are already de-duplicated upstream.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional

from .alert import Alert
from .anomaly import RiskAssessment
from .social_distance import SocialDistanceResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    camera_id TEXT NOT NULL,
    count INTEGER,
    risk_score REAL,
    risk_level TEXT,
    predicted_level TEXT,
    sd_violations INTEGER
);
CREATE INDEX IF NOT EXISTS idx_frames_ts ON frames (camera_id, ts);

CREATE TABLE IF NOT EXISTS alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    camera_id TEXT NOT NULL,
    level TEXT,
    score REAL,
    predicted_level TEXT,
    reasons TEXT
);
CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts (camera_id, ts);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    camera_id TEXT NOT NULL,
    event_type TEXT,
    confidence REAL,
    description TEXT
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events (camera_id, ts);
"""


@dataclass
class HistoryStore:
    db_path: str = "crowd_monitor_history.db"
    record_every_seconds: float = 1.0

    def __post_init__(self) -> None:
        self._last_frame_ts: dict[str, float] = {}
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=5)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def record_frame(
        self,
        camera_id: str,
        count: int,
        risk: RiskAssessment,
        social_distance: SocialDistanceResult,
        ts: Optional[float] = None,
    ) -> bool:
        ts = ts if ts is not None else time.time()
        last = self._last_frame_ts.get(camera_id, float("-inf"))
        if ts - last < self.record_every_seconds:
            return False
        self._last_frame_ts[camera_id] = ts
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO frames (ts, camera_id, count, risk_score, risk_level, "
                "predicted_level, sd_violations) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (ts, camera_id, count, risk.score, risk.level, risk.predicted_level, social_distance.violation_count),
            )
        return True

    def record_alert(self, camera_id: str, alert: Alert) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO alerts (ts, camera_id, level, score, predicted_level, reasons) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (alert.timestamp, camera_id, alert.level, alert.score, alert.predicted_level, json.dumps(alert.reasons)),
            )

    def record_event(
        self, camera_id: str, event_type: str, confidence: float, description: str = "", ts: Optional[float] = None
    ) -> None:
        ts = ts if ts is not None else time.time()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO events (ts, camera_id, event_type, confidence, description) VALUES (?, ?, ?, ?, ?)",
                (ts, camera_id, event_type, confidence, description),
            )

    def recent_alerts(self, camera_id: Optional[str] = None, limit: int = 50) -> list[dict]:
        query = "SELECT ts, camera_id, level, score, predicted_level, reasons FROM alerts"
        params: tuple = ()
        if camera_id is not None:
            query += " WHERE camera_id = ?"
            params = (camera_id,)
        query += " ORDER BY ts DESC LIMIT ?"
        params = params + (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {
                "ts": r[0], "camera_id": r[1], "level": r[2], "score": r[3],
                "predicted_level": r[4], "reasons": json.loads(r[5]),
            }
            for r in rows
        ]

    def recent_events(self, camera_id: Optional[str] = None, limit: int = 50) -> list[dict]:
        query = "SELECT ts, camera_id, event_type, confidence, description FROM events"
        params: tuple = ()
        if camera_id is not None:
            query += " WHERE camera_id = ?"
            params = (camera_id,)
        query += " ORDER BY ts DESC LIMIT ?"
        params = params + (limit,)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [
            {"ts": r[0], "camera_id": r[1], "event_type": r[2], "confidence": r[3], "description": r[4]}
            for r in rows
        ]

    def summary_since(self, since_ts: float, camera_id: Optional[str] = None) -> dict:
        """Peak count/score and per-level alert counts since `since_ts` — the
        kind of roll-up a shift-handover report or a daily digest would want."""
        frame_query = "SELECT MAX(count), MAX(risk_score) FROM frames WHERE ts >= ?"
        frame_params: tuple = (since_ts,)
        alert_query = "SELECT level, COUNT(*) FROM alerts WHERE ts >= ?"
        alert_params: tuple = (since_ts,)
        if camera_id is not None:
            frame_query += " AND camera_id = ?"
            frame_params = frame_params + (camera_id,)
            alert_query += " AND camera_id = ?"
            alert_params = alert_params + (camera_id,)
        alert_query += " GROUP BY level"

        with self._connect() as conn:
            peak_count, peak_score = conn.execute(frame_query, frame_params).fetchone()
            level_counts = dict(conn.execute(alert_query, alert_params).fetchall())

        return {
            "peak_count": peak_count or 0,
            "peak_score": peak_score or 0.0,
            "alert_counts_by_level": level_counts,
        }
