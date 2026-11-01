from crowd_monitor.density import DensityEstimator
from crowd_monitor.detector import Detection


def test_empty_frame_is_sparse():
    estimator = DensityEstimator(rows=4, cols=4)
    result = estimator.compute([], frame_shape=(360, 480))
    assert result.count == 0
    assert result.level == "sparse"
    assert result.peak_cell_count == 0


def test_clustered_detections_raise_density_level():
    estimator = DensityEstimator(rows=4, cols=4, moderate_threshold=2, dense_threshold=4, critical_threshold=7)
    # 8 detections all centered in the same small region -> one grid cell
    detections = [Detection(x=200 + i, y=150 + i, w=10, h=10) for i in range(8)]
    result = estimator.compute(detections, frame_shape=(360, 480))
    assert result.count == 8
    assert result.peak_cell_count == 8
    assert result.level == "critical"


def test_spread_out_detections_stay_sparse():
    estimator = DensityEstimator(rows=4, cols=4, moderate_threshold=2, dense_threshold=4, critical_threshold=7)
    detections = [Detection(x=x, y=50, w=10, h=10) for x in (10, 130, 250, 370)]
    result = estimator.compute(detections, frame_shape=(360, 480))
    assert result.count == 4
    assert result.peak_cell_count == 1
    assert result.level == "sparse"
