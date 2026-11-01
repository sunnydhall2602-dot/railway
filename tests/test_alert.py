import time

from crowd_monitor.alert import AlertManager
from crowd_monitor.anomaly import RiskAssessment
from crowd_monitor.hazard_detector import HazardDetection


def _risk(level, score=0.0):
    return RiskAssessment(score=score, level=level, predicted_score=score, predicted_level=level, reasons=[])


def _hazard(label="fire", confidence=0.9):
    return HazardDetection(label=label, confidence=confidence, x=0, y=0, w=10, h=10)


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


def test_hazard_fires_critical_alert_regardless_of_crowd_risk():
    manager = AlertManager(camera_id="cam-test")
    alert = manager.process_hazard([_hazard("fire")])
    assert alert is not None
    assert alert.level == "CRITICAL"
    assert "fire" in alert.reasons[0]


def test_no_hazard_means_no_alert():
    manager = AlertManager(camera_id="cam-test")
    assert manager.process_hazard([]) is None


def test_hazard_alert_respects_its_own_cooldown():
    manager = AlertManager(camera_id="cam-test", hazard_repeat_seconds=3600)
    manager.process_hazard([_hazard("smoke")])
    second = manager.process_hazard([_hazard("smoke")])
    assert second is None


def test_hazard_cooldown_resets_once_hazard_clears():
    manager = AlertManager(camera_id="cam-test", hazard_repeat_seconds=3600)
    manager.process_hazard([_hazard("fire")])
    manager.process_hazard([])  # hazard clears -> cooldown resets
    second = manager.process_hazard([_hazard("fire")])
    assert second is not None
