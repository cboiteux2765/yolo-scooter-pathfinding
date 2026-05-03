from scooter_open_path_guidance_app.guidance_core import GuidancePlanner, TrackedObstacle


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
