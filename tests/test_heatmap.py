import numpy as np

from crowd_monitor.detector import Detection
from crowd_monitor.heatmap import DwellHeatmap


def test_empty_frame_produces_no_overlay_change():
    heatmap = DwellHeatmap()
    frame = np.zeros((100, 120, 3), dtype=np.uint8)
    heatmap.update([], frame_shape=(100, 120))
    out = heatmap.overlay(frame)
    assert np.array_equal(out, frame)


def test_repeated_occupancy_accumulates_heat():
    heatmap = DwellHeatmap(decay=0.9, radius=5, gain=1.0)
    frame_shape = (100, 120)
    detections = [Detection(x=55, y=45, w=10, h=10)]  # centroid ~(60, 50)

    for _ in range(30):
        heat = heatmap.update(detections, frame_shape)

    assert heat[50, 60] > 0
    # heat should have grown with repeated occupancy, not stayed flat
    heat_after_more = heatmap.update(detections, frame_shape)
    assert heat_after_more[50, 60] >= heat[50, 60] * 0.95


def test_passerby_decays_away():
    heatmap = DwellHeatmap(decay=0.5, radius=5, gain=1.0)
    frame_shape = (100, 120)
    heatmap.update([Detection(x=55, y=45, w=10, h=10)], frame_shape)
    # nobody there afterwards -> heat should decay toward zero
    for _ in range(20):
        heat = heatmap.update([], frame_shape)
    assert heat[50, 60] < 0.01


def test_hotspots_returns_requested_count():
    heatmap = DwellHeatmap()
    frame_shape = (120, 160)
    detections = [Detection(x=10, y=10, w=10, h=10)]
    for _ in range(10):
        heatmap.update(detections, frame_shape)
    spots = heatmap.hotspots(rows=4, cols=4, top_n=2)
    assert len(spots) == 2
    # sorted descending by heat
    assert spots[0][2] >= spots[1][2]
