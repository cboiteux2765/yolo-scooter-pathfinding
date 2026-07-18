from scooter_open_path_guidance_app.guidance_core import (
    GuidanceDecision,
    GuidancePlanner,
    TrackedObstacle,
    build_instruction_text,
)


def make_obstacle(x: float, y: float, radius: float, vx: float = 0.0) -> TrackedObstacle:
    return TrackedObstacle(
        label="person",
        bbox=(x - radius, y - radius, x + radius, y + radius),
        center=(x, y),
        radius_px=radius,
        vx=vx,
    )


def test_prefers_open_center_gap() -> None:
    planner = GuidancePlanner()
    obstacles = [
        make_obstacle(110, 380, 65),
        make_obstacle(540, 380, 65),
    ]
    decision = planner.plan(640, 480, obstacles)
    assert decision.command in {"slow_down", "maintain_speed", "speed_up"}
    assert abs(decision.corridor_center_x - 320) < 60


def test_slows_for_dense_path() -> None:
    planner = GuidancePlanner()
    obstacles = [
        make_obstacle(120, 370, 75),
        make_obstacle(250, 360, 75),
        make_obstacle(390, 355, 75, vx=80),
        make_obstacle(520, 360, 75),
    ]
    decision = planner.plan(640, 480, obstacles)
    assert decision.command in {"slow_down", "wait", "stop"}
    assert decision.risk_score > 0.5


def test_stops_when_no_safe_gap_exists() -> None:
    planner = GuidancePlanner()
    obstacles = [
        make_obstacle(90, 400, 110),
        make_obstacle(250, 400, 110),
        make_obstacle(410, 400, 110),
        make_obstacle(570, 400, 110),
    ]
    decision = planner.plan(640, 480, obstacles)
    assert decision.command == "stop"


def test_build_instruction_text_for_center_hold() -> None:
    decision = GuidanceDecision(
        command="maintain_speed",
        speed_factor=1.0,
        risk_score=0.1,
        corridor_center_x=320,
        corridor_width_px=180,
        recommended_heading_px=0.0,
        band_top=200,
        band_bottom=420,
    )
    assert build_instruction_text(decision, 640) == "Maintain speed and hold center."


def test_build_instruction_text_for_slow_right_turn() -> None:
    decision = GuidanceDecision(
        command="slow_down",
        speed_factor=0.45,
        risk_score=0.7,
        corridor_center_x=460,
        corridor_width_px=120,
        recommended_heading_px=110.0,
        band_top=200,
        band_bottom=420,
    )
    assert build_instruction_text(decision, 640) == "Slow down and move slightly right."
