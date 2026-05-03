# Scooter Open Path Guidance App

This repository is now a pure Python starter for real-time scooter guidance from monocular video using Ultralytics and OpenCV.

It is intentionally a prototype, not an autonomous control stack. The current code is aimed at helping you validate perception, motion tracking, rider clearance, and phone-delivered guidance before you invest in a full 3D stack.

## What Is Implemented

- `scooter_open_path_guidance_app/app.py`
  OpenCV video runner for webcam, file, or RTSP input
- `scooter_open_path_guidance_app/guidance_core.py`
  heuristic planner that inflates obstacles, predicts motion, finds open horizontal gaps, and emits `maintain_speed`, `slow_down`, `speed_up`, `wait`, or `stop`
- `scooter_open_path_guidance_app/http_api.py`
  built-in HTTP endpoint for polling the latest guidance payload from a phone or Streamlit client
- `scooter_open_path_guidance_app/export_yolo26n.py`
  helper to export a YOLO checkpoint to ONNX
- `configs/guidance.yml`
  tuning knobs for detector filters, scooter width, planning band, and risk thresholds
- `tests/`
  planner unit tests
- `docs/prototype_plan.md`
  exact end-to-end build plan, current model recommendation, and dataset guidance

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
  the local process can publish the latest command as JSON

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

The polling endpoint is:

```text
http://127.0.0.1:8765/latest
```

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

## Proof that it works
Create new venv, activate, install requirements.txt

```python -m pytest -q``` 3/3 passed

```python -m scooter_open_path_guidance_app.app 0 --api```

[Result of running that command](https://drive.google.com/file/d/1obi9hkzwzZjT19MC1gbhqvFTGJTk4LBG/view?usp=drive_link)