# Scooter Pathfinding Prototype Plan

## Model choice

As of May 3, 2026, Ultralytics' official docs describe `YOLO26` as the latest model family and still recommend both `YOLO26` and `YOLO11` for stable production workloads. For this prototype, `yolo26n.pt` is the most cost-efficient default because Ultralytics currently reports slightly better COCO mAP than `yolo11n.pt` with a smaller parameter count and better CPU latency.

- `YOLO26n`: 40.9 mAP, 38.9 ms CPU ONNX, 1.7 ms T4 TensorRT, 2.4M params
- `YOLO11n`: 39.5 mAP, 56.1 ms CPU ONNX, 1.5 ms T4 TensorRT, 2.6M params

Recommendation:

1. Start with `YOLO26n` for the first end-to-end prototype.
2. Keep `YOLO11n` as a fallback if you run into model-specific tooling friction.
3. Do not treat `YOLO12` as a "smaller" option. It is a different model family, and an Ultralytics maintainer discussion notes slower training and higher memory use than `YOLO11`.

## Dataset stack

Use a layered dataset strategy instead of looking for one perfect dataset.

1. `COCO`
   Use for pretrained weights and broad person/car/bicycle/dog/cat coverage. This is the fastest way to get a baseline detector running.
2. `BDD100K`
   Use for road scenes, traffic context, drivable area, and tracking-oriented fine-tuning.
3. `KITTI`
   Use for stereo, visual odometry, depth, 3D object detection, and 3D tracking experiments.
4. `nuScenes`
   Use when you want richer multi-sensor urban driving scenes and stronger trajectory context.
5. `Waymo Open Dataset`
   Use for later-stage 2D/3D perception and tracking benchmarking once you want a larger-scale autonomy-grade dataset.

Important gap:

Driving datasets are good for road context, but they are not the best source for roadside animals. For animal-heavy safety cases, keep COCO pretraining, then add a small custom roadside-animal dataset from your own clips or curated public footage.

## What this repo now implements

This repository now contains a runnable monocular prototype with:

1. Ultralytics tracking on webcam, video, or RTSP input
2. Circle-based obstacle inflation that can merge `person + bicycle/motorcycle` into a larger rider obstacle
3. Motion-aware risk scoring using tracked velocity
4. Open-gap corridor planning
5. Commands: `maintain_speed`, `slow_down`, `speed_up`, `wait`, `stop`
6. A pseudo-3D illuminated path overlay for visual debugging
7. A tiny polling API for a phone or Streamlit client

## Exact end-to-end build steps

### Phase 1: Baseline perception

1. Create a Python environment and install `scooter_open_path_guidance_app/requirements.txt`.
2. Run the app with `yolo26n.pt` on a webcam or short recorded ride.
3. Verify detections for `person`, `bicycle`, `motorcycle`, `car`, `bus`, `truck`, `dog`, and `cat`.
4. Tune `configs/guidance.yml` so the inflated rider circles match realistic scooter clearance.

### Phase 2: Motion-aware guidance

1. Record several short path videos with pedestrians crossing, curves, and narrow sidewalks.
2. Review the overlay video and note false `speed_up` or late `slow_down` cases.
3. Adjust:
   - `base_safety_radius_px`
   - `extra_person_scooter_radius_px`
   - `prediction_horizon_s`
   - `occupancy_slowdown_threshold`
   - `blind_corner_edge_fraction`
4. Save representative failure clips for future regression testing.

### Phase 3: Real 3D capability

The current overlay is perspective-aware, but it is not metric 3D. To make the system truly 3D:

1. Calibrate the camera and store intrinsics.
2. Add one of these depth approaches:
   - stereo camera
   - RGB-D camera
   - monocular depth model
3. Transform each tracked obstacle from image space into approximate ground-plane coordinates.
4. Replace image-space circle inflation with metric-space occupancy discs or boxes.
5. Use time-to-collision instead of only pixel clearance for `slow_down` and `stop`.

### Phase 4: Dataset fine-tuning

1. Keep COCO pretrained weights as the starting checkpoint.
2. Fine-tune on BDD100K for road context and path occupancy.
3. Add KITTI or nuScenes if you want depth or 3D evaluation.
4. Add a custom `scooter`, `stroller`, and `roadside_animal` class if those are critical.
5. Benchmark `YOLO26n`, `YOLO26s`, and `YOLO11n` on your actual target hardware before moving up in size.

### Phase 5: Phone voice guidance

1. Run the local CV process with `--api`.
2. Expose the JSON payload through your local network, tunnel, or relay service.
3. Have the Streamlit app poll `/latest`.
4. Convert command changes into phone speech:
   - `slow_down` -> "Slow down, low visibility ahead."
   - `speed_up` -> "Path opening up, you can speed up slightly."
   - `wait` -> "Hold here, crossing motion ahead."
   - `stop` -> "Stop now."
5. Add hysteresis so repeated identical commands do not spam the rider.
