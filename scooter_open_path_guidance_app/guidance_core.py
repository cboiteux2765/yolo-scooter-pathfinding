from __future__ import annotations

from dataclasses import asdict, dataclass, field
from math import hypot
from typing import Iterable


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class TrackedObstacle:
    label: str
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    radius_px: float
    confidence: float = 1.0
    track_id: int | None = None
    vx: float = 0.0
    vy: float = 0.0
    depth_m: float | None = None

    def predicted_center(self, horizon_s: float) -> tuple[float, float]:
        return (
            self.center[0] + self.vx * horizon_s,
            self.center[1] + self.vy * horizon_s,
        )


@dataclass
class GapCandidate:
    left: float
    right: float
    score: float

    @property
    def center_x(self) -> float:
        return (self.left + self.right) * 0.5

    @property
    def width(self) -> float:
        return self.right - self.left


@dataclass
class GuidanceDecision:
    command: str
    speed_factor: float
    risk_score: float
    corridor_center_x: float
    corridor_width_px: float
    recommended_heading_px: float
    band_top: int
    band_bottom: int
    reasons: list[str] = field(default_factory=list)
    path_points: list[tuple[int, int]] = field(default_factory=list)
    gap_segments: list[tuple[int, int]] = field(default_factory=list)

    def to_payload(self) -> dict:
        return asdict(self)


class GuidancePlanner:
    """Heuristic path planner that turns tracked obstacles into speed guidance."""

    def __init__(self, config: dict | None = None) -> None:
        cfg = config or {}
        planner_cfg = cfg.get("planner", cfg)
        self.top_fraction = planner_cfg.get("planning_band", {}).get("top_fraction", 0.42)
        self.bottom_fraction = planner_cfg.get("planning_band", {}).get("bottom_fraction", 0.92)
        self.base_safety_radius_px = float(planner_cfg.get("base_safety_radius_px", 22))
        self.scooter_width_fraction = float(planner_cfg.get("scooter_width_fraction", 0.16))
        self.min_gap_width_fraction = float(planner_cfg.get("min_gap_width_fraction", 0.28))
        self.emergency_stop_distance_fraction = float(
            planner_cfg.get("emergency_stop_distance_fraction", 0.16)
        )
        self.slow_down_distance_fraction = float(
            planner_cfg.get("slow_down_distance_fraction", 0.34)
        )
        self.prediction_horizon_s = float(planner_cfg.get("prediction_horizon_s", 0.9))
        self.occupancy_slowdown_threshold = float(
            planner_cfg.get("occupancy_slowdown_threshold", 0.42)
        )
        self.speedup_gap_width_fraction = float(
            planner_cfg.get("speedup_gap_width_fraction", 0.45)
        )
        self.center_preference_weight = float(
            planner_cfg.get("center_preference_weight", 0.30)
        )
        self.blind_corner_edge_fraction = float(planner_cfg.get("blind_corner_edge_fraction", 0.15))
        self.path_samples = int(planner_cfg.get("path_samples", 9))

    def plan(
        self,
        frame_width: int,
        frame_height: int,
        obstacles: Iterable[TrackedObstacle],
    ) -> GuidanceDecision:
        obstacles = list(obstacles)
        band_top = int(frame_height * self.top_fraction)
        band_bottom = int(frame_height * self.bottom_fraction)
        band_height = max(1, band_bottom - band_top)
        scooter_width_px = frame_width * self.scooter_width_fraction
        min_gap_px = frame_width * self.min_gap_width_fraction

        blocked_segments: list[tuple[float, float]] = []
        active_obstacles: list[tuple[TrackedObstacle, tuple[float, float]]] = []

        for obstacle in obstacles:
            predicted_x, predicted_y = obstacle.predicted_center(self.prediction_horizon_s)
            inflated_radius = obstacle.radius_px + self.base_safety_radius_px + scooter_width_px * 0.5
            if predicted_y + inflated_radius < band_top or predicted_y - inflated_radius > band_bottom:
                continue
            left = clamp(predicted_x - inflated_radius, 0.0, float(frame_width))
            right = clamp(predicted_x + inflated_radius, 0.0, float(frame_width))
            blocked_segments.append((left, right))
            active_obstacles.append((obstacle, (predicted_x, predicted_y)))

        merged_segments = self._merge_segments(blocked_segments)
        gaps = self._compute_gaps(frame_width, merged_segments, min_gap_px)
        chosen_gap = self._choose_gap(frame_width, gaps)

        occupancy = self._occupancy_ratio(frame_width, merged_segments)
        clearance_fraction = self._clearance_fraction(
            band_bottom,
            band_height,
            active_obstacles,
        )
        crossing_risk = self._crossing_risk(frame_width, active_obstacles, chosen_gap)
        blind_corner_risk = self._blind_corner_risk(frame_width, active_obstacles, chosen_gap)

        reasons: list[str] = []
        if occupancy > self.occupancy_slowdown_threshold:
            reasons.append("dense_path")
        if clearance_fraction < self.slow_down_distance_fraction:
            reasons.append("close_obstacle")
        if crossing_risk > 0.45:
            reasons.append("crossing_motion")
        if blind_corner_risk > 0.45:
            reasons.append("low_visibility_corner")

        distance_risk = 1.0 - clearance_fraction
        occupancy_risk = clamp(occupancy / max(self.occupancy_slowdown_threshold, 1e-6), 0.0, 1.0)
        risk_score = clamp(
            0.42 * distance_risk + 0.28 * occupancy_risk + 0.20 * crossing_risk + 0.10 * blind_corner_risk,
            0.0,
            1.0,
        )

        center_x = float(frame_width) * 0.5
        gap_segments = [(int(g.left), int(g.right)) for g in gaps]

        if chosen_gap is None:
            command = "stop"
            corridor_center_x = center_x
            corridor_width_px = scooter_width_px
            reasons.append("no_safe_gap")
        else:
            corridor_center_x = chosen_gap.center_x
            corridor_width_px = max(scooter_width_px, chosen_gap.width * 0.7)
            if clearance_fraction < self.emergency_stop_distance_fraction or chosen_gap.width < scooter_width_px:
                command = "stop"
                reasons.append("blocked_ahead")
            elif risk_score > 0.82:
                command = "wait"
            elif risk_score > 0.56 or chosen_gap.width < min_gap_px * 1.15:
                command = "slow_down"
            elif (
                risk_score < 0.24
                and chosen_gap.width > frame_width * self.speedup_gap_width_fraction
                and occupancy < 0.18
            ):
                command = "speed_up"
            else:
                command = "maintain_speed"

        speed_factor_map = {
            "stop": 0.0,
            "wait": 0.10,
            "slow_down": 0.45,
            "maintain_speed": 1.0,
            "speed_up": 1.2,
        }
        path_points = self._build_path_points(
            frame_width,
            frame_height,
            band_top,
            band_bottom,
            center_x,
            corridor_center_x,
        )
        return GuidanceDecision(
            command=command,
            speed_factor=speed_factor_map[command],
            risk_score=risk_score,
            corridor_center_x=corridor_center_x,
            corridor_width_px=corridor_width_px,
            recommended_heading_px=corridor_center_x - center_x,
            band_top=band_top,
            band_bottom=band_bottom,
            reasons=reasons,
            path_points=path_points,
            gap_segments=gap_segments,
        )

    @staticmethod
    def _merge_segments(segments: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if not segments:
            return []
        ordered = sorted(segments)
        merged = [ordered[0]]
        for left, right in ordered[1:]:
            prev_left, prev_right = merged[-1]
            if left <= prev_right:
                merged[-1] = (prev_left, max(prev_right, right))
            else:
                merged.append((left, right))
        return merged

    def _compute_gaps(
        self,
        frame_width: int,
        merged_segments: list[tuple[float, float]],
        min_gap_px: float,
    ) -> list[GapCandidate]:
        gaps: list[GapCandidate] = []
        cursor = 0.0
        center_x = frame_width * 0.5
        for left, right in merged_segments:
            if left > cursor:
                score = self._gap_score(cursor, left, center_x, frame_width)
                gaps.append(GapCandidate(cursor, left, score))
            cursor = max(cursor, right)
        if cursor < frame_width:
            score = self._gap_score(cursor, float(frame_width), center_x, frame_width)
            gaps.append(GapCandidate(cursor, float(frame_width), score))
        return [gap for gap in gaps if gap.width >= min_gap_px * 0.8]

    def _gap_score(self, left: float, right: float, center_x: float, frame_width: int) -> float:
        width_score = (right - left) / max(float(frame_width), 1.0)
        center_penalty = abs(((left + right) * 0.5) - center_x) / max(center_x, 1.0)
        return width_score - center_penalty * self.center_preference_weight

    @staticmethod
    def _choose_gap(frame_width: int, gaps: list[GapCandidate]) -> GapCandidate | None:
        if not gaps:
            return None
        center_x = frame_width * 0.5
        return max(
            gaps,
            key=lambda gap: (gap.score, -abs(gap.center_x - center_x), gap.width),
        )

    @staticmethod
    def _occupancy_ratio(frame_width: int, merged_segments: list[tuple[float, float]]) -> float:
        occupied = sum(max(0.0, right - left) for left, right in merged_segments)
        return clamp(occupied / max(float(frame_width), 1.0), 0.0, 1.0)

    @staticmethod
    def _clearance_fraction(
        band_bottom: int,
        band_height: int,
        active_obstacles: list[tuple[TrackedObstacle, tuple[float, float]]],
    ) -> float:
        if not active_obstacles:
            return 1.0
        closest = min(
            max(0.0, band_bottom - (predicted_y - obstacle.radius_px))
            for obstacle, (_, predicted_y) in active_obstacles
        )
        return clamp(closest / max(float(band_height), 1.0), 0.0, 1.0)

    def _crossing_risk(
        self,
        frame_width: int,
        active_obstacles: list[tuple[TrackedObstacle, tuple[float, float]]],
        chosen_gap: GapCandidate | None,
    ) -> float:
        if not active_obstacles or chosen_gap is None:
            return 0.0
        risk = 0.0
        target_x = chosen_gap.center_x
        for obstacle, (predicted_x, _) in active_obstacles:
            motion_mag = hypot(obstacle.vx, obstacle.vy)
            if motion_mag < 5.0:
                continue
            moving_toward_gap = abs(target_x - (predicted_x + obstacle.vx * 0.4)) < abs(
                target_x - predicted_x
            )
            if moving_toward_gap:
                norm_speed = clamp(motion_mag / max(frame_width * 0.35, 1.0), 0.0, 1.0)
                risk = max(risk, norm_speed)
        return risk

    def _blind_corner_risk(
        self,
        frame_width: int,
        active_obstacles: list[tuple[TrackedObstacle, tuple[float, float]]],
        chosen_gap: GapCandidate | None,
    ) -> float:
        if chosen_gap is None:
            return 1.0
        edge_threshold = frame_width * self.blind_corner_edge_fraction
        gap_center = chosen_gap.center_x
        edge_bias = 0.0
        if gap_center < edge_threshold or gap_center > frame_width - edge_threshold:
            edge_bias = 0.55
        occluder_bias = 0.0
        for obstacle, (predicted_x, _) in active_obstacles:
            if predicted_x < edge_threshold or predicted_x > frame_width - edge_threshold:
                occluder_bias = max(
                    occluder_bias,
                    clamp(obstacle.radius_px / max(frame_width * 0.25, 1.0), 0.0, 0.45),
                )
        return clamp(edge_bias + occluder_bias, 0.0, 1.0)

    def _build_path_points(
        self,
        frame_width: int,
        frame_height: int,
        band_top: int,
        band_bottom: int,
        start_x: float,
        target_x: float,
    ) -> list[tuple[int, int]]:
        path: list[tuple[int, int]] = []
        start_y = frame_height - 1
        for index in range(self.path_samples):
            t = index / max(self.path_samples - 1, 1)
            smooth = t * t * (3.0 - 2.0 * t)
            y = int(round(start_y + (band_top - start_y) * smooth))
            mid_pull = 0.28 if band_bottom != band_top else 0.0
            x = start_x + (target_x - start_x) * (smooth * (1.0 - mid_pull) + t * mid_pull)
            path.append((int(round(x)), y))
        return path


def describe_heading(recommended_heading_px: float, frame_width: int) -> str:
    if frame_width <= 0:
        return "center"
    normalized = recommended_heading_px / max(frame_width * 0.5, 1.0)
    magnitude = abs(normalized)
    if magnitude < 0.08:
        return "center"
    direction = "left" if normalized < 0.0 else "right"
    if magnitude < 0.36:
        return f"slightly {direction}"
    return direction


def build_instruction_text(decision: GuidanceDecision, frame_width: int) -> str:
    heading_text = describe_heading(decision.recommended_heading_px, frame_width)
    if decision.command == "stop":
        return "Stop now."
    if decision.command == "wait":
        return "Wait and hold position."
    if decision.command == "slow_down":
        if heading_text == "center":
            return "Slow down and hold center."
        return f"Slow down and move {heading_text}."
    if decision.command == "speed_up":
        if heading_text == "center":
            return "Path is clear. Speed up and hold center."
        return f"Path is clear. Speed up and move {heading_text}."
    if heading_text == "center":
        return "Maintain speed and hold center."
    return f"Maintain speed and move {heading_text}."
