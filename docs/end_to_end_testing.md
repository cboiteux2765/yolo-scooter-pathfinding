# End-to-End Real-World Testing

This project is an advisory prototype, so the safest real-world test is a staged rollout instead of immediately riding in live traffic.

## 1. Bench Test Indoors

Use a webcam or recorded clip first:

```bash
python -m scooter_open_path_guidance_app.app 0 --api --speak --otel --otel-exporter console
```

What to verify:

- the OpenCV window updates continuously
- spoken guidance changes when people or bikes cross the frame
- console spans appear for `guidance.capture`, `guidance.inference`, `guidance.planning`, and `guidance.overlay`
- console metrics include frame latency, detections per frame, obstacles per frame, and command counts

## 2. Low-Risk Outdoor Walk Test

Mount the camera on the scooter, but walk beside it instead of riding.

Goals:

- verify the camera angle still shows the planning band clearly
- verify the app can keep up with outdoor lighting and shadows
- compare spoken instructions against what a human observer would say
- confirm no command chatter happens when the scene is stable

Capture evidence:

- save an annotated video with `--output`
- record a phone video of the scene and the spoken guidance
- note any frames where the instruction was clearly late or wrong

## 3. Closed-Course Ride Test

Only move to this stage after the walk test looks stable.

Recommended setup:

- empty parking lot or closed path
- one rider
- one spotter
- staged obstacles such as cones, a walking person, and a parked bike

Run:

```bash
python -m scooter_open_path_guidance_app.app 0 --output runs/closed-course.mp4 --speak --otel --otel-exporter console
```

What to score manually:

- did `stop` happen before a human would brake
- did `slow_down` appear in dense scenes
- did left/right suggestions match the visible open gap
- did instructions remain understandable at riding speed

## 4. Collector-Backed Telemetry Test

If you want traces and metrics outside the terminal, point the app at an OTLP/HTTP collector.

Example app run:

```bash
python -m scooter_open_path_guidance_app.app 0 --otel --otel-exporter otlp_http --otel-endpoint http://127.0.0.1:4318
```

What to verify in your backend:

- one `guidance.frame` trace per processed frame
- child spans for capture, inference, planning, overlay, publish, and speech queueing
- histograms for frame latency and per-stage latency
- counters for processed frames, capture failures, and emitted commands

## 5. Pass/Fail Criteria

Before treating the prototype as useful, aim for all of these:

- median frame latency is comfortably below your decision budget
- instructions are stable and not oscillating frame to frame
- the planner does not repeatedly suggest unsafe openings in your closed-course test
- missed detections are explainable and fixable with model/data changes
- guidance remains usable across morning, noon, and evening lighting

## 6. What “Working” Means Here

For this repo, end-to-end success means:

- OpenCV captures frames reliably
- YOLO tracking runs in real time on your hardware
- the planner emits reasonable commands
- voice or phone surfaces reflect those commands quickly
- OpenTelemetry confirms where latency and failures occur

That is enough to validate the prototype loop. It is not enough to claim the system is safe for autonomous riding.
