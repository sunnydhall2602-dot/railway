import numpy as np
import pytest

from crowd_monitor.hazard_detector import HazardDetector


def test_disabled_without_weights_path():
    detector = HazardDetector()
    assert detector.enabled is False
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert detector.detect(frame) == []


def test_bad_weights_path_raises_clear_error():
    with pytest.raises(RuntimeError, match="Could not load hazard model"):
        HazardDetector(weights_path="/nonexistent/model.pt")
