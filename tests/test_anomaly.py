import numpy as np

from crowd_monitor.anomaly import TemporalAnomalyScorer
from crowd_monitor.density import DensityResult
from crowd_monitor.flow import FlowResult


def _density(level="sparse", peak=0):
    return DensityResult(count=peak, grid=np.zeros((4, 4)), peak_cell_count=peak, peak_cell=(0, 0), level=level)


def test_calm_scene_is_low_risk():
    scorer = TemporalAnomalyScorer(fps=15)
    result = scorer.update(_density(), FlowResult(mean_speed=0.1, divergence=0.0, angular_variance=0.01))
    assert result.level == "LOW"
    assert result.score < 25


def test_panic_signature_is_high_risk():
    scorer = TemporalAnomalyScorer(fps=15)
    panic_flow = FlowResult(mean_speed=3.0, divergence=0.08, angular_variance=0.8)
    result = scorer.update(_density(level="critical", peak=8), panic_flow)
    assert result.level in ("HIGH", "CRITICAL")
    assert result.score >= 50


def test_predicted_score_extrapolates_rising_trend():
    scorer = TemporalAnomalyScorer(fps=15, window_seconds=2, predict_ahead_seconds=10)
    calm = FlowResult(mean_speed=0.1, divergence=0.0, angular_variance=0.0)
    rising_panic = FlowResult(mean_speed=2.0, divergence=0.05, angular_variance=0.6)

    for _ in range(10):
        scorer.update(_density(), calm)
    last_result = None
    for _ in range(10):
        last_result = scorer.update(_density(level="dense", peak=5), rising_panic)

    assert last_result is not None
    # the trend is rising, so the model should predict further escalation
    # ahead of the current score, not just echo the current frame back.
    assert last_result.predicted_score >= last_result.score
