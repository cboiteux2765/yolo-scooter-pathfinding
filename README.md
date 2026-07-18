# Scooter Open Path Guidance App

This repository is now a pure Python starter for real-time scooter guidance from monocular video using Ultralytics and OpenCV.

It is intentionally a prototype, not an autonomous control stack. The current code is aimed at helping you validate perception, motion tracking, rider clearance, and phone-delivered guidance before you invest in a full 3D stack.

## What Is Implemented

- `scooter_open_path_guidance_app/app.py`
  OpenCV video runner for webcam, file, or RTSP input
- `scooter_open_path_guidance_app/guidance_core.py`
  heuristic planner that inflates obstacles, predicts motion, finds open horizontal gaps, and emits `maintain_speed`, `slow_down`, `speed_up`, `wait`, or `stop`
- `scooter_open_path_guidance_app/http_api.py`
  built-in HTTP endpoint plus live mobile dashboard for polling the latest guidance payload and annotated frame from a phone browser
- `scooter_open_path_guidance_app/export_yolo26n.py`
  helper to export a YOLO checkpoint to ONNX
- `configs/guidance.yml`
  tuning knobs for detector filters, scooter width, planning band, and risk thresholds
- `tests/`
  planner unit tests
- `docs/prototype_plan.md`
  exact end-to-end build plan, current model recommendation, and dataset guidance
- `scooter_open_path_guidance_app/evaluate.py`
  evaluation entry point for detector metrics and end-to-end guidance accuracy
- `docs/metrics.md`
  how to measure mAP, command accuracy, confusion matrix, and guidance regression quality
- `docs/end_to_end_testing.md`
  staged real-world validation checklist for bench, walk, and closed-course testing

## Prototype Features

- one input stream at a time
- Ultralytics detection + tracking through the Python API
- rider-aware obstacle inflation
  if a `person` overlaps a `bicycle` or `motorcycle`, the app fuses them into a larger rider obstacle and draws a larger safety circle
- motion-aware risk scoring
  tracked velocity can raise the risk score for crossing pedestrians and crowded paths
- dynamic illuminated corridor overlay
  the planner draws a pseudo-3D safe path for debugging and demo purposes
- optional phone/Streamlit bridge
  the local process can publish the latest command as JSON and a live phone dashboard
- browser voice guidance
  the phone dashboard can speak short guidance instructions such as stop, slow down, or move slightly left
- local-first spoken guidance
  the Python runner can speak instructions directly on the inference device so voice is not dependent on Wi-Fi or phone polling
- OpenTelemetry instrumentation
  optional traces and metrics around OpenCV capture, YOLO inference, planning, overlay rendering, publishing, and voice queueing

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r scooter_open_path_guidance_app/requirements.txt
```

## Run

Webcam:

```bash
python -m scooter_open_path_guidance_app.app 0
```

Video file:

```bash
python -m scooter_open_path_guidance_app.app path/to/video.mp4
```

CUDA if available:

```bash
python -m scooter_open_path_guidance_app.app 0 --device cuda:0
```

Write an annotated output video:

```bash
python -m scooter_open_path_guidance_app.app path/to/video.mp4 --output runs/annotated.mp4
```

Publish the latest guidance decision for a phone client:

```bash
python -m scooter_open_path_guidance_app.app 0 --api
```

Open the live dashboard in a browser on the same computer:

```text
http://127.0.0.1:8765/
```

The raw polling endpoint is:

```text
http://127.0.0.1:8765/latest
```

Use it from your phone on the same Wi-Fi network:

```bash
python -m scooter_open_path_guidance_app.app 0 --api --api-host 0.0.0.0
```

Then open:

```text
http://<your-computer-ip>:8765/
```

The phone page shows:

- the latest annotated frame
- the current command and risk
- the reasons behind the command
- a one-tap voice guidance button that uses the phone browser's speech engine

For the lowest-latency setup, keep guidance local to the device running inference:

```bash
python -m scooter_open_path_guidance_app.app 0 --speak
```

That path keeps:

- YOLO inference local
- path planning local
- spoken instructions local

Use the phone dashboard only as an optional observer surface, not as the critical real-time guidance path.

## Telemetry

There are two telemetry layers in this repo:

- app telemetry
  the live JSON payload published at `/latest` already contains command, risk, heading, and obstacle state for the latest frame
- OpenTelemetry
  optional traces and metrics now instrument the OpenCV loop itself so you can see where latency and failures come from

Enable console export:

```bash
python -m scooter_open_path_guidance_app.app 0 --otel --otel-exporter console
```

Enable OTLP/HTTP export to a collector:

```bash
python -m scooter_open_path_guidance_app.app 0 --otel --otel-exporter otlp_http --otel-endpoint http://127.0.0.1:4318
```

The OpenTelemetry integration records:

- one `guidance.frame` span per processed frame
- child spans for capture, inference, planning, overlay, publish, and speech queueing
- counters for processed frames, capture failures, and emitted commands
- histograms for frame latency, stage latency, detections per frame, obstacles per frame, and risk score

This instrumentation is manual and code-based. That follows the OpenTelemetry Python guidance for manual instrumentation with the SDK and API, and OTLP endpoint configuration through environment or exporter setup.

## Tuning

Edit `configs/guidance.yml` to tune:

- obstacle labels considered by the planner
- planning band bounds
- safety radius around each obstacle
- assumed scooter width
- minimum acceptable gap width
- stop and slowdown thresholds
- prediction horizon for tracked motion

## 3D Reality Check

This starter does not claim true metric 3D yet.

What it does now:

- image-space obstacle circles
- motion prediction from 2D tracks
- perspective-style corridor rendering

What you still need for real 3D guidance:

- calibrated camera intrinsics
- depth estimation or stereo
- ground-plane projection
- time-to-collision or metric clearance logic

## Model Recommendation

For the first prototype, start with `yolo26n.pt`.

Why:

- it is the current latest Ultralytics family
- it is fast enough for real-time prototyping
- it gives you a better CPU-latency baseline than `yolo11n` in Ultralytics' current published tables

More detail, including the current `YOLO26` vs `YOLO11` vs `YOLO12` recommendation and dataset choices, is in `docs/prototype_plan.md`.

## Notes

- This is an advisory CV system, not autonomous control.
- The path overlay is a debug aid and demo surface, not a navigation guarantee.
- The most valuable next step is adding real depth and a small custom scooter safety dataset.

## Metrics

The repository now supports two kinds of evaluation:

- detector metrics with Ultralytics validation
  measures `mAP50`, `mAP50-95`, precision, recall, and per-class performance on a labeled detection dataset
- end-to-end guidance metrics on labeled video frames
  measures command accuracy, macro precision/recall/F1, confusion matrix, and optional risk/heading error

Detector evaluation:

```bash
python -m scooter_open_path_guidance_app.evaluate detect --model yolo26n.pt --data path/to/data.yaml --split val
```

Guidance evaluation:

```bash
python -m scooter_open_path_guidance_app.evaluate guidance path/to/video.mp4 --annotations path/to/guidance_labels.json
```

More detail is in `docs/metrics.md`.

## End-to-End Testing

The safest way to test this in real life is staged:

1. Indoor bench test with a webcam or saved clip.
2. Outdoor walk test with the scooter while you stay off it.
3. Closed-course ride test in an empty lot or path.
4. Telemetry review to confirm where latency and bad commands came from.

The detailed checklist is in `docs/end_to_end_testing.md`.

## Proof that it works
Create new venv, activate, install requirements.txt

```python -m pytest -q``` 3/3 passed

```python -m scooter_open_path_guidance_app.app 0 --api```

[Result of running that command](https://drive.google.com/file/d/1obi9hkzwzZjT19MC1gbhqvFTGJTk4LBG/view?usp=drive_link)
