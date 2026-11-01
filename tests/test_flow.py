import numpy as np

from crowd_monitor.flow import FlowAnalyzer


def _frame_with_square(width: int, height: int, x: int, y: int, size: int = 30) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[y : y + size, x : x + size] = 255
    return frame


def test_first_frame_returns_zero_result():
    analyzer = FlowAnalyzer(resize_width=160)
    result = analyzer.compute(_frame_with_square(320, 240, 50, 50))
    assert result.mean_speed == 0.0
    assert result.divergence == 0.0


def test_static_scene_has_near_zero_speed():
    analyzer = FlowAnalyzer(resize_width=160)
    frame = _frame_with_square(320, 240, 50, 50)
    analyzer.compute(frame)
    result = analyzer.compute(frame)
    assert result.mean_speed == 0.0


def test_moving_object_produces_motion_signal():
    analyzer = FlowAnalyzer(resize_width=160)
    analyzer.compute(_frame_with_square(320, 240, 50, 50))
    result = analyzer.compute(_frame_with_square(320, 240, 65, 50))
    assert result.mean_speed > 0.0


def test_reset_clears_previous_frame():
    analyzer = FlowAnalyzer(resize_width=160)
    analyzer.compute(_frame_with_square(320, 240, 50, 50))
    analyzer.reset()
    result = analyzer.compute(_frame_with_square(320, 240, 65, 50))
    assert result.mean_speed == 0.0  # no previous frame to compare against
