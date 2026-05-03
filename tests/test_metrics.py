from scooter_open_path_guidance_app.metrics import (
    compute_command_metrics,
    compute_detection_metrics,
)


def test_compute_command_metrics() -> None:
    expected = [
        {"command": "slow_down", "risk_score": 0.9, "heading_px": -10},
        {"command": "maintain_speed", "risk_score": 0.2, "heading_px": 5},
        {"command": "stop", "risk_score": 1.0, "heading_px": 0},
    ]
    predicted = [
        {"command": "slow_down", "risk_score": 0.8, "heading_px": -2},
        {"command": "speed_up", "risk_score": 0.3, "heading_px": 11},
        {"command": "stop", "risk_score": 0.95, "heading_px": 4},
    ]

    metrics = compute_command_metrics(expected, predicted)

    assert metrics.total_frames == 3
    assert metrics.correct_frames == 2
    assert round(metrics.accuracy, 4) == round(2 / 3, 4)
    assert metrics.risk_mae is not None
    assert metrics.heading_mae_px is not None
    assert "slow_down" in metrics.per_class


def test_compute_detection_metrics() -> None:
    ground_truth = [
        {"label": "person", "bbox": (10, 10, 50, 80)},
        {"label": "car", "bbox": (100, 20, 220, 120)},
    ]
    predicted = [
        {"label": "person", "bbox": (12, 12, 48, 82)},
        {"label": "car", "bbox": (102, 18, 224, 119)},
        {"label": "dog", "bbox": (260, 40, 310, 100)},
    ]

    metrics = compute_detection_metrics(ground_truth, predicted, iou_threshold=0.5)

    assert metrics.matched_predictions == 2
    assert metrics.false_positives == 1
    assert metrics.false_negatives == 0
    assert metrics.precision < 1.0
    assert metrics.recall == 1.0
