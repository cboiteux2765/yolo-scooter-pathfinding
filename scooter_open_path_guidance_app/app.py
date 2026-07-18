from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

from .guidance_core import GuidancePlanner, TrackedObstacle, build_instruction_text
from .http_api import GuidancePublisher
from .speech import InstructionSpeaker


PERSON_LABELS = {"person"}
RIDER_SUPPORT_LABELS = {"bicycle", "motorcycle", "scooter"}


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-time scooter open path guidance prototype")
    parser.add_argument("source", help="webcam index, video path, or RTSP URL")
    parser.add_argument("--config", default="configs/guidance.yml")
    parser.add_argument("--model", default=None, help="override model checkpoint, e.g. yolo26n.pt")
    parser.add_argument("--device", default=None, help="cpu, cuda:0, mps, etc.")
    parser.add_argument("--output", default=None, help="optional annotated video path")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--iou", type=float, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--tracker", default=None, help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--api", action="store_true", help="publish latest guidance at /latest")
    parser.add_argument("--api-host", default=None, help="HTTP host, use 0.0.0.0 for phone access")
    parser.add_argument("--api-port", type=int, default=None, help="HTTP port for the mobile dashboard")
    parser.add_argument("--speak", action="store_true", help="speak guidance locally on the inference device")
    parser.add_argument("--speech-rate", type=int, default=185, help="words per minute for local speech")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=None, help="useful for short test runs")
    return parser.parse_args()


def resolve_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def resolve_class_filter(model: YOLO, labels: list[str] | None) -> list[int] | None:
    if not labels:
        return None
    names = model.names
    name_to_id = {str(name): idx for idx, name in names.items()} if isinstance(names, dict) else {}
    class_ids = [name_to_id[label] for label in labels if label in name_to_id]
    return class_ids or None


def extract_raw_detections(result: Any) -> list[dict[str, Any]]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []
    names = result.names
    ids = None
    if getattr(boxes, "id", None) is not None:
        ids = boxes.id.int().cpu().tolist()
    xyxy = boxes.xyxy.cpu().tolist()
    confs = boxes.conf.cpu().tolist()
    classes = boxes.cls.int().cpu().tolist()

    detections: list[dict[str, Any]] = []
    for index, bbox in enumerate(xyxy):
        x1, y1, x2, y2 = bbox
        detections.append(
            {
                "label": str(names[int(classes[index])]),
                "bbox": (float(x1), float(y1), float(x2), float(y2)),
                "confidence": float(confs[index]),
                "track_id": int(ids[index]) if ids is not None else None,
            }
        )
    return detections


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


def union_bbox(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (
        min(a[0], b[0]),
        min(a[1], b[1]),
        max(a[2], b[2]),
        max(a[3], b[3]),
    )


def estimate_velocity(
    track_memory: dict[int, dict[str, float]],
    track_id: int | None,
    center: tuple[float, float],
    now_s: float,
    alpha: float,
) -> tuple[float, float]:
    if track_id is None:
        return 0.0, 0.0
    state = track_memory.get(track_id)
    if state is None:
        track_memory[track_id] = {"x": center[0], "y": center[1], "t": now_s, "vx": 0.0, "vy": 0.0}
        return 0.0, 0.0
    dt = max(now_s - state["t"], 1e-3)
    raw_vx = (center[0] - state["x"]) / dt
    raw_vy = (center[1] - state["y"]) / dt
    vx = state["vx"] * alpha + raw_vx * (1.0 - alpha)
    vy = state["vy"] * alpha + raw_vy * (1.0 - alpha)
    track_memory[track_id] = {"x": center[0], "y": center[1], "t": now_s, "vx": vx, "vy": vy}
    return vx, vy


def prune_tracks(track_memory: dict[int, dict[str, float]], now_s: float, stale_after_s: float = 2.0) -> None:
    stale_ids = [track_id for track_id, state in track_memory.items() if now_s - state["t"] > stale_after_s]
    for track_id in stale_ids:
        track_memory.pop(track_id, None)


def convert_to_obstacles(
    detections: list[dict[str, Any]],
    frame_width: int,
    track_memory: dict[int, dict[str, float]],
    now_s: float,
    velocity_alpha: float,
    extra_person_scooter_radius_px: float,
) -> list[TrackedObstacle]:
    used: set[int] = set()
    people = [idx for idx, det in enumerate(detections) if det["label"] in PERSON_LABELS]
    riders = [idx for idx, det in enumerate(detections) if det["label"] in RIDER_SUPPORT_LABELS]
    obstacles: list[TrackedObstacle] = []

    for person_idx in people:
        if person_idx in used:
            continue
        person = detections[person_idx]
        matched_idx = None
        for rider_idx in riders:
            if rider_idx in used:
                continue
            rider = detections[rider_idx]
            iou = bbox_iou(person["bbox"], rider["bbox"])
            if iou > 0.03:
                matched_idx = rider_idx
                break
        if matched_idx is None:
            continue
        rider = detections[matched_idx]
        used.add(person_idx)
        used.add(matched_idx)
        bbox = union_bbox(person["bbox"], rider["bbox"])
        center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        radius_px = max(width, height) * 0.5 + extra_person_scooter_radius_px
        track_id = person["track_id"] if person["track_id"] is not None else rider["track_id"]
        vx, vy = estimate_velocity(track_memory, track_id, center, now_s, velocity_alpha)
        obstacles.append(
            TrackedObstacle(
                label="rider",
                bbox=bbox,
                center=center,
                radius_px=radius_px,
                confidence=max(person["confidence"], rider["confidence"]),
                track_id=track_id,
                vx=vx,
                vy=vy,
            )
        )

    for index, detection in enumerate(detections):
        if index in used:
            continue
        bbox = detection["bbox"]
        center = ((bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        label = detection["label"]
        radius_px = max(width, height) * 0.5
        if label in PERSON_LABELS | RIDER_SUPPORT_LABELS:
            radius_px += extra_person_scooter_radius_px
        vx, vy = estimate_velocity(track_memory, detection["track_id"], center, now_s, velocity_alpha)
        obstacles.append(
            TrackedObstacle(
                label=label,
                bbox=bbox,
                center=center,
                radius_px=radius_px,
                confidence=detection["confidence"],
                track_id=detection["track_id"],
                vx=vx,
                vy=vy,
            )
        )

    prune_tracks(track_memory, now_s)
    return obstacles


def draw_guidance_overlay(frame: Any, decision: Any, obstacles: list[TrackedObstacle], show_band: bool, show_debug_text: bool) -> Any:
    annotated = frame.copy()
    overlay = annotated.copy()
    color_map = {
        "stop": (0, 0, 255),
        "wait": (0, 90, 255),
        "slow_down": (0, 220, 255),
        "maintain_speed": (0, 200, 80),
        "speed_up": (255, 180, 0),
    }
    path_color = color_map[decision.command]

    if show_band:
        cv2.rectangle(
            overlay,
            (0, decision.band_top),
            (annotated.shape[1] - 1, decision.band_bottom),
            (80, 80, 80),
            -1,
        )
        cv2.addWeighted(overlay, 0.08, annotated, 0.92, 0.0, annotated)

    for obstacle in obstacles:
        x1, y1, x2, y2 = map(int, obstacle.bbox)
        cx, cy = map(int, obstacle.center)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (230, 230, 230), 1)
        cv2.circle(annotated, (cx, cy), int(obstacle.radius_px), (0, 140, 255), 2)
        cv2.arrowedLine(
            annotated,
            (cx, cy),
            (int(cx + obstacle.vx * 0.12), int(cy + obstacle.vy * 0.12)),
            (255, 255, 0),
            2,
            tipLength=0.25,
        )
        label = obstacle.label if obstacle.track_id is None else f"{obstacle.label}#{obstacle.track_id}"
        cv2.putText(
            annotated,
            label,
            (x1, max(14, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    if len(decision.path_points) >= 2:
        left_rail: list[tuple[int, int]] = []
        right_rail: list[tuple[int, int]] = []
        total = max(len(decision.path_points) - 1, 1)
        for index, (x, y) in enumerate(decision.path_points):
            width_scale = 1.0 - (index / total) * 0.72
            half_width = max(12, int(decision.corridor_width_px * 0.5 * width_scale))
            left_rail.append((x - half_width, y))
            right_rail.append((x + half_width, y))
        polygon = left_rail + list(reversed(right_rail))
        polygon_array = np.array(polygon, dtype=np.int32)
        cv2.fillPoly(overlay, [polygon_array], path_color)
        cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0.0, annotated)
        cv2.polylines(annotated, [np.array(left_rail, dtype=np.int32)], False, path_color, 2)
        cv2.polylines(annotated, [np.array(right_rail, dtype=np.int32)], False, path_color, 2)
        cv2.polylines(
            annotated,
            [np.array(decision.path_points, dtype=np.int32)],
            False,
            (255, 255, 255),
            2,
        )

    for left, right in decision.gap_segments:
        cv2.line(annotated, (left, decision.band_top), (right, decision.band_top), (100, 255, 100), 2)

    if show_debug_text:
        hud_lines = [
            f"command: {decision.command}",
            f"risk: {decision.risk_score:.2f}",
            f"heading_px: {decision.recommended_heading_px:.0f}",
            f"reasons: {', '.join(decision.reasons) if decision.reasons else 'clear'}",
        ]
        for index, text in enumerate(hud_lines):
            cv2.putText(
                annotated,
                text,
                (12, 24 + index * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )

    return annotated


def ensure_writer(path: str, capture: cv2.VideoCapture, frame_shape: tuple[int, int, int]) -> cv2.VideoWriter:
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 20.0
    width = frame_shape[1]
    height = frame_shape[0]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(path, fourcc, fps, (width, height))


def encode_frame_jpeg(frame: Any, quality: int = 75) -> bytes | None:
    ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return encoded.tobytes()


def discover_lan_ip() -> str | None:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.connect(("8.8.8.8", 80))
            return str(probe.getsockname()[0])
    except OSError:
        return None


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config))
    model_cfg = config.get("model", {})
    overlay_cfg = config.get("overlay", {})
    api_cfg = config.get("api", {})
    planner = GuidancePlanner(config)

    model_name = args.model or model_cfg.get("name", "yolo26n.pt")
    conf = args.conf if args.conf is not None else float(model_cfg.get("conf", 0.25))
    iou = args.iou if args.iou is not None else float(model_cfg.get("iou", 0.45))
    imgsz = args.imgsz if args.imgsz is not None else int(model_cfg.get("imgsz", 640))
    tracker = args.tracker or model_cfg.get("tracker", "bytetrack.yaml")
    class_filter = None

    model = YOLO(model_name)
    class_filter = resolve_class_filter(model, model_cfg.get("class_labels"))

    publisher = None
    speaker = None
    if args.api:
        api_host = args.api_host or str(api_cfg.get("host", "127.0.0.1"))
        api_port = args.api_port or int(api_cfg.get("port", 8765))
        publisher = GuidancePublisher(
            host=api_host,
            port=api_port,
        )
        publisher.start()
        announced_host = api_host
        if api_host == "0.0.0.0":
            announced_host = discover_lan_ip() or "<your-computer-ip>"
        print(f"Phone dashboard: http://{announced_host}:{api_port}/")
    if args.speak:
        speaker = InstructionSpeaker(rate=args.speech_rate)

    capture = cv2.VideoCapture(resolve_source(args.source))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source: {args.source}")

    output_writer = None
    track_memory: dict[int, dict[str, float]] = {}
    frame_index = 0
    velocity_alpha = float(config.get("planner", {}).get("track_velocity_alpha", 0.65))
    rider_radius_boost = float(config.get("planner", {}).get("extra_person_scooter_radius_px", 16))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

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
            now_s = time.time()
            obstacles = convert_to_obstacles(
                detections,
                frame.shape[1],
                track_memory,
                now_s,
                velocity_alpha,
                rider_radius_boost,
            )
            decision = planner.plan(frame.shape[1], frame.shape[0], obstacles)
            instruction_text = build_instruction_text(decision, frame.shape[1])
            annotated = draw_guidance_overlay(
                frame,
                decision,
                obstacles,
                bool(overlay_cfg.get("show_planning_band", True)),
                bool(overlay_cfg.get("show_debug_text", True)),
            )

            payload = {
                "timestamp": now_s,
                "frame_index": frame_index,
                "command": decision.command,
                "speed_factor": decision.speed_factor,
                "risk_score": decision.risk_score,
                "recommended_heading_px": decision.recommended_heading_px,
                "reasons": decision.reasons,
                "instruction_text": instruction_text,
                "voice_instruction": instruction_text,
                "decision": decision.to_payload(),
                "obstacles": [
                    {
                        "label": obstacle.label,
                        "track_id": obstacle.track_id,
                        "center": obstacle.center,
                        "radius_px": obstacle.radius_px,
                        "vx": obstacle.vx,
                        "vy": obstacle.vy,
                    }
                    for obstacle in obstacles
                ],
            }
            if publisher is not None:
                publisher.update(payload, frame_jpeg=encode_frame_jpeg(annotated))
            if speaker is not None:
                speaker.submit(instruction_text)

            if args.output:
                if output_writer is None:
                    output_writer = ensure_writer(args.output, capture, annotated.shape)
                output_writer.write(annotated)

            if not args.no_display:
                cv2.imshow("Scooter Open Path Guidance", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord("q")}:
                    break
                if key == ord("j"):
                    print(json.dumps(payload, indent=2))

            frame_index += 1
            if args.max_frames is not None and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if output_writer is not None:
            output_writer.release()
        if publisher is not None:
            publisher.stop()
        if speaker is not None:
            speaker.close()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
