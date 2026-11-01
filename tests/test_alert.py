import time

from crowd_monitor.alert import AlertManager
from crowd_monitor.anomaly import RiskAssessment


def _risk(level, score=0.0):
    return RiskAssessment(score=score, level=level, predicted_score=score, predicted_level=level, reasons=[])


def test_escalation_triggers_immediate_alert():
    manager = AlertManager(camera_id="cam-test")
    assert manager.process(_risk("LOW")) is None
    alert = manager.process(_risk("HIGH", score=60))
    assert alert is not None
    assert alert.level == "HIGH"


def test_sustained_same_level_does_not_spam():
    manager = AlertManager(camera_id="cam-test", repeat_seconds=3600)
    manager.process(_risk("HIGH", score=60))
    second = manager.process(_risk("HIGH", score=61))
    assert second is None  # still HIGH, no new escalation, cooldown not elapsed


def test_sustained_risk_eventually_re_notifies():
    manager = AlertManager(camera_id="cam-test", repeat_seconds=0.01)
    manager.process(_risk("HIGH", score=60))
    time.sleep(0.02)
    second = manager.process(_risk("HIGH", score=61))
    assert second is not None


def test_de_escalation_is_not_alerted():
    manager = AlertManager(camera_id="cam-test", repeat_seconds=3600)
    manager.process(_risk("HIGH", score=60))
    result = manager.process(_risk("LOW", score=5))
    assert result is None
