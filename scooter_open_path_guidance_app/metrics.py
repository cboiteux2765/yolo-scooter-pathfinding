from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def bbox_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0


@dataclass
class CommandMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    total_frames: int
    correct_frames: int
    confusion_matrix: dict[str, dict[str, int]]
    per_class: dict[str, dict[str, float]]
    risk_mae: float | None = None
    heading_mae_px: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "macro_precision": self.macro_precision,
            "macro_recall": self.macro_recall,
            "macro_f1": self.macro_f1,
            "total_frames": self.total_frames,
            "correct_frames": self.correct_frames,
            "confusion_matrix": self.confusion_matrix,
            "per_class": self.per_class,
            "risk_mae": self.risk_mae,
            "heading_mae_px": self.heading_mae_px,
        }


@dataclass
class DetectionMetrics:
    precision: float
    recall: float
    f1: float
    matched_predictions: int
    false_positives: int
    false_negatives: int
    total_predictions: int
    total_ground_truth: int
    mean_iou: float
    per_class: dict[str, dict[str, float]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "matched_predictions": self.matched_predictions,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "total_predictions": self.total_predictions,
            "total_ground_truth": self.total_ground_truth,
            "mean_iou": self.mean_iou,
            "per_class": self.per_class,
        }


def compute_command_metrics(
    expected: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
) -> CommandMetrics:
    if len(expected) != len(predicted):
        raise ValueError("Expected and predicted command lists must have the same length.")

    labels = sorted(
        {
            str(item["command"])
            for item in expected + predicted
            if item.get("command") is not None
        }
    )
    confusion = {
        expected_label: {predicted_label: 0 for predicted_label in labels}
        for expected_label in labels
    }

    correct = 0
    risk_errors: list[float] = []
    heading_errors: list[float] = []

    for truth, pred in zip(expected, predicted):
        truth_label = str(truth["command"])
        pred_label = str(pred["command"])
        confusion[truth_label][pred_label] += 1
        if truth_label == pred_label:
            correct += 1
        if truth.get("risk_score") is not None and pred.get("risk_score") is not None:
            risk_errors.append(abs(float(truth["risk_score"]) - float(pred["risk_score"])))
        if truth.get("heading_px") is not None and pred.get("heading_px") is not None:
            heading_errors.append(abs(float(truth["heading_px"]) - float(pred["heading_px"])))

    per_class: dict[str, dict[str, float]] = {}
    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        precision = safe_divide(tp, tp + fp)
        recall = safe_divide(tp, tp + fn)
        f1 = safe_divide(2 * precision * recall, precision + recall)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }

    total = len(expected)
    return CommandMetrics(
        accuracy=safe_divide(correct, total),
        macro_precision=safe_divide(sum(precisions), len(precisions)),
        macro_recall=safe_divide(sum(recalls), len(recalls)),
        macro_f1=safe_divide(sum(f1s), len(f1s)),
        total_frames=total,
        correct_frames=correct,
        confusion_matrix=confusion,
        per_class=per_class,
        risk_mae=safe_divide(sum(risk_errors), len(risk_errors)) if risk_errors else None,
        heading_mae_px=safe_divide(sum(heading_errors), len(heading_errors)) if heading_errors else None,
    )


def compute_detection_metrics(
    ground_truth: list[dict[str, Any]],
    predicted: list[dict[str, Any]],
    iou_threshold: float = 0.5,
) -> DetectionMetrics:
    matched_pred_indices: set[int] = set()
    matched_gt_indices: set[int] = set()
    ious: list[float] = []
    labels = sorted(
        {
            str(item["label"])
            for item in ground_truth + predicted
            if item.get("label") is not None
        }
    )

    per_label_counts: dict[str, dict[str, int]] = {
        label: {"tp": 0, "fp": 0, "fn": 0} for label in labels
    }

    for gt_index, gt_item in enumerate(ground_truth):
        best_index = None
        best_iou = 0.0
        for pred_index, pred_item in enumerate(predicted):
            if pred_index in matched_pred_indices:
                continue
            if str(pred_item["label"]) != str(gt_item["label"]):
                continue
            overlap = bbox_iou(tuple(gt_item["bbox"]), tuple(pred_item["bbox"]))
            if overlap >= iou_threshold and overlap > best_iou:
                best_iou = overlap
                best_index = pred_index
        if best_index is not None:
            matched_gt_indices.add(gt_index)
            matched_pred_indices.add(best_index)
            ious.append(best_iou)
            per_label_counts[str(gt_item["label"])]["tp"] += 1

    for gt_index, gt_item in enumerate(ground_truth):
        if gt_index not in matched_gt_indices:
            per_label_counts[str(gt_item["label"])]["fn"] += 1

    for pred_index, pred_item in enumerate(predicted):
        if pred_index not in matched_pred_indices:
            per_label_counts[str(pred_item["label"])]["fp"] += 1

    matched = len(matched_gt_indices)
    false_positives = len(predicted) - matched
    false_negatives = len(ground_truth) - matched
    precision = safe_divide(matched, len(predicted))
    recall = safe_divide(matched, len(ground_truth))
    f1 = safe_divide(2 * precision * recall, precision + recall)

    per_class: dict[str, dict[str, float]] = {}
    for label, counts in per_label_counts.items():
        label_precision = safe_divide(counts["tp"], counts["tp"] + counts["fp"])
        label_recall = safe_divide(counts["tp"], counts["tp"] + counts["fn"])
        label_f1 = safe_divide(2 * label_precision * label_recall, label_precision + label_recall)
        per_class[label] = {
            "precision": label_precision,
            "recall": label_recall,
            "f1": label_f1,
            "support": counts["tp"] + counts["fn"],
        }

    return DetectionMetrics(
        precision=precision,
        recall=recall,
        f1=f1,
        matched_predictions=matched,
        false_positives=false_positives,
        false_negatives=false_negatives,
        total_predictions=len(predicted),
        total_ground_truth=len(ground_truth),
        mean_iou=safe_divide(sum(ious), len(ious)),
        per_class=per_class,
    )


def summarize_ultralytics_val(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    names = getattr(metrics, "names", {})
    per_class: dict[str, dict[str, float]] = {}

    if box is not None and hasattr(box, "maps"):
        maps = list(box.maps)
        for index, map_value in enumerate(maps):
            label = str(names.get(index, index)) if isinstance(names, dict) else str(index)
            per_class[label] = {"map50_95": float(map_value)}

    return {
        "map50": float(getattr(box, "map50", 0.0)) if box is not None else 0.0,
        "map50_95": float(getattr(box, "map", 0.0)) if box is not None else 0.0,
        "precision": float(getattr(box, "mp", 0.0)) if box is not None else 0.0,
        "recall": float(getattr(box, "mr", 0.0)) if box is not None else 0.0,
        "fitness": float(getattr(metrics, "fitness", 0.0)),
        "per_class": per_class,
    }
