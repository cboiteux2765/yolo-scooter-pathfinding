from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from .app import (
    convert_to_obstacles,
    extract_raw_detections,
    load_config,
    resolve_class_filter,
    resolve_source,
)
from .guidance_core import GuidancePlanner
from .metrics import compute_command_metrics, summarize_ultralytics_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate detector and guidance accuracy.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    detect_parser = subparsers.add_parser("detect", help="Run Ultralytics validation on a labeled detection dataset.")
    detect_parser.add_argument("--model", default="yolo26n.pt")
    detect_parser.add_argument("--data", required=True, help="Ultralytics dataset YAML path.")
    detect_parser.add_argument("--split", default="val", help="Dataset split, usually val or test.")
    detect_parser.add_argument("--imgsz", type=int, default=640)
    detect_parser.add_argument("--conf", type=float, default=0.25)
    detect_parser.add_argument("--iou", type=float, default=0.45)
    detect_parser.add_argument("--device", default=None)
    detect_parser.add_argument("--save-json", default=None, help="Optional path to save metric JSON.")

    guidance_parser = subparsers.add_parser(
        "guidance",
        help="Run end-to-end guidance evaluation on a labeled video.",
    )
    guidance_parser.add_argument("source", help="video path, webcam index, or RTSP URL")
    guidance_parser.add_argument("--annotations", required=True, help="JSON file with frame-level expected commands.")
    guidance_parser.add_argument("--config", default="configs/guidance.yml")
    guidance_parser.add_argument("--model", default=None)
    guidance_parser.add_argument("--device", default=None)
    guidance_parser.add_argument("--conf", type=float, default=None)
    guidance_parser.add_argument("--iou", type=float, default=None)
    guidance_parser.add_argument("--imgsz", type=int, default=None)
    guidance_parser.add_argument("--tracker", default=None)
    guidance_parser.add_argument("--save-json", default=None, help="Optional path to save metric JSON.")
    return parser.parse_args()


def save_json_if_requested(payload: dict[str, Any], output_path: str | None) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def print_json(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, indent=2))


def run_detection_eval(args: argparse.Namespace) -> dict[str, Any]:
    model = YOLO(args.model)
    result = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        verbose=False,
    )
    payload = {
        "mode": "detect",
        "model": args.model,
        "data": args.data,
        "split": args.split,
        "metrics": summarize_ultralytics_val(result),
    }
    save_json_if_requested(payload, args.save_json)
    return payload


def load_guidance_annotations(path: Path) -> dict[int, dict[str, Any]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    frames = raw.get("frames")
    if not isinstance(frames, list):
        raise ValueError("Annotations JSON must contain a top-level 'frames' list.")

    indexed: dict[int, dict[str, Any]] = {}
    for item in frames:
        frame_index = int(item["frame_index"])
        if "command" not in item:
            raise ValueError(f"Annotation for frame {frame_index} is missing 'command'.")
        indexed[frame_index] = item
    return indexed


def run_guidance_eval(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(Path(args.config))
    planner = GuidancePlanner(config)
    model_cfg = config.get("model", {})

    model_name = args.model or model_cfg.get("name", "yolo26n.pt")
    conf = args.conf if args.conf is not None else float(model_cfg.get("conf", 0.25))
    iou = args.iou if args.iou is not None else float(model_cfg.get("iou", 0.45))
    imgsz = args.imgsz if args.imgsz is not None else int(model_cfg.get("imgsz", 640))
    tracker = args.tracker or model_cfg.get("tracker", "bytetrack.yaml")
    velocity_alpha = float(config.get("planner", {}).get("track_velocity_alpha", 0.65))
    rider_radius_boost = float(config.get("planner", {}).get("extra_person_scooter_radius_px", 16))

    annotation_map = load_guidance_annotations(Path(args.annotations))
    model = YOLO(model_name)
    class_filter = resolve_class_filter(model, model_cfg.get("class_labels"))

    capture = cv2.VideoCapture(resolve_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    expected_rows: list[dict[str, Any]] = []
    predicted_rows: list[dict[str, Any]] = []
    track_memory: dict[int, dict[str, float]] = {}
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index not in annotation_map:
                frame_index += 1
                continue

            results = model.track(
                frame,
                persist=True,
                verbose=False,
                conf=conf,
                iou=iou,
                imgsz=imgsz,
                device=args.device,
                tracker=tracker,
                classes=class_filter,
            )
            result = results[0]
            detections = extract_raw_detections(result)
            obstacles = convert_to_obstacles(
                detections,
                frame.shape[1],
                track_memory,
                frame_index / max(capture.get(cv2.CAP_PROP_FPS), 1.0),
                velocity_alpha,
                rider_radius_boost,
            )
            decision = planner.plan(frame.shape[1], frame.shape[0], obstacles)
            annotation = annotation_map[frame_index]

            expected_rows.append(
                {
                    "command": annotation["command"],
                    "risk_score": annotation.get("risk_score"),
                    "heading_px": annotation.get("heading_px"),
                }
            )
            predicted_rows.append(
                {
                    "command": decision.command,
                    "risk_score": decision.risk_score,
                    "heading_px": decision.recommended_heading_px,
                }
            )
            frame_index += 1
    finally:
        capture.release()

    metrics = compute_command_metrics(expected_rows, predicted_rows)
    payload = {
        "mode": "guidance",
        "model": model_name,
        "source": args.source,
        "annotations": args.annotations,
        "labeled_frames": len(expected_rows),
        "metrics": metrics.to_dict(),
    }
    save_json_if_requested(payload, args.save_json)
    return payload


def main() -> None:
    args = parse_args()
    if args.mode == "detect":
        payload = run_detection_eval(args)
    else:
        payload = run_guidance_eval(args)
    print_json(payload)


if __name__ == "__main__":
    main()
