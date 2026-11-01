import json

from crowd_monitor.alert import Alert
from crowd_monitor.anomaly import RiskAssessment
from crowd_monitor.history import HistoryStore
from crowd_monitor.social_distance import SocialDistanceResult


def _risk(level="LOW", score=10.0):
    return RiskAssessment(score=score, level=level, predicted_score=score, predicted_level=level, reasons=[])


def test_record_frame_is_throttled(tmp_path):
    store = HistoryStore(db_path=str(tmp_path / "history.db"), record_every_seconds=1.0)
    sd = SocialDistanceResult()
    assert store.record_frame("cam-01", 5, _risk(), sd, ts=0.0) is True
    assert store.record_frame("cam-01", 5, _risk(), sd, ts=0.3) is False  # too soon
    assert store.record_frame("cam-01", 5, _risk(), sd, ts=1.1) is True


def test_record_and_query_alerts(tmp_path):
    store = HistoryStore(db_path=str(tmp_path / "history.db"))
    alert = Alert(camera_id="cam-01", level="HIGH", score=60.0, predicted_level="CRITICAL",
                  reasons=["fast crowd movement"], timestamp=100.0)
    store.record_alert("cam-01", alert)
    rows = store.recent_alerts(camera_id="cam-01")
    assert len(rows) == 1
    assert rows[0]["level"] == "HIGH"
    assert rows[0]["reasons"] == ["fast crowd movement"]


def test_record_and_query_events(tmp_path):
    store = HistoryStore(db_path=str(tmp_path / "history.db"))
    store.record_event("cam-01", "fire", 0.92, "hazard detected", ts=50.0)
    rows = store.recent_events(camera_id="cam-01")
    assert len(rows) == 1
    assert rows[0]["event_type"] == "fire"
    assert rows[0]["confidence"] == 0.92


def test_summary_since(tmp_path):
    store = HistoryStore(db_path=str(tmp_path / "history.db"), record_every_seconds=0)
    sd = SocialDistanceResult()
    store.record_frame("cam-01", 10, _risk(score=40.0), sd, ts=10.0)
    store.record_frame("cam-01", 20, _risk(score=80.0), sd, ts=20.0)
    store.record_alert("cam-01", Alert("cam-01", "HIGH", 80.0, "CRITICAL", [], timestamp=20.0))
    store.record_alert("cam-01", Alert("cam-01", "MEDIUM", 40.0, "HIGH", [], timestamp=11.0))

    summary = store.summary_since(0.0, camera_id="cam-01")
    assert summary["peak_count"] == 20
    assert summary["peak_score"] == 80.0
    assert summary["alert_counts_by_level"] == {"HIGH": 1, "MEDIUM": 1}


def test_persists_across_instances(tmp_path):
    db_path = str(tmp_path / "history.db")
    store1 = HistoryStore(db_path=db_path)
    store1.record_event("cam-01", "smoke", 0.7, ts=5.0)

    store2 = HistoryStore(db_path=db_path)
    rows = store2.recent_events()
    assert len(rows) == 1
    assert rows[0]["event_type"] == "smoke"
