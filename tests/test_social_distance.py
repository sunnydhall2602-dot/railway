from crowd_monitor.detector import Detection
from crowd_monitor.social_distance import SocialDistanceChecker


def test_no_violations_with_fewer_than_two_people():
    checker = SocialDistanceChecker()
    assert checker.check([]).violation_count == 0
    assert checker.check([Detection(0, 0, 20, 40)]).violation_count == 0


def test_close_pair_is_flagged():
    checker = SocialDistanceChecker(min_distance_px=60, use_dynamic_scale=False)
    detections = [Detection(x=100, y=100, w=20, h=40), Detection(x=110, y=100, w=20, h=40)]
    result = checker.check(detections)
    assert result.violation_count == 1
    assert result.people_in_violation == 2
    assert result.violation_ratio == 1.0


def test_far_pair_is_not_flagged():
    checker = SocialDistanceChecker(min_distance_px=60, use_dynamic_scale=False)
    detections = [Detection(x=0, y=0, w=20, h=40), Detection(x=500, y=500, w=20, h=40)]
    result = checker.check(detections)
    assert result.violation_count == 0
    assert result.violation_ratio == 0.0


def test_dynamic_scale_widens_threshold_for_larger_detections():
    checker = SocialDistanceChecker(min_distance_px=10, use_dynamic_scale=True, scale_factor=1.0)
    # detections are tall (height 200) -> dynamic threshold (~200px) exceeds the 70px gap
    detections = [Detection(x=0, y=0, w=50, h=200), Detection(x=70, y=0, w=50, h=200)]
    result = checker.check(detections)
    assert result.threshold_px >= 200
    assert result.violation_count == 1


def test_three_people_mixed_distances():
    checker = SocialDistanceChecker(min_distance_px=50, use_dynamic_scale=False)
    detections = [
        Detection(x=0, y=0, w=10, h=10),    # centroid (5,5)
        Detection(x=10, y=0, w=10, h=10),   # centroid (15,5) -> close to first
        Detection(x=1000, y=1000, w=10, h=10),  # far from both
    ]
    result = checker.check(detections)
    assert result.violation_count == 1
    assert result.people_in_violation == 2
    assert round(result.violation_ratio, 3) == round(2 / 3, 3)
