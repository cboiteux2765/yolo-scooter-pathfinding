# Accuracy Metrics

This project now supports two separate evaluation layers:

1. detector accuracy
2. end-to-end guidance accuracy

You want both. A detector can have good mAP and still produce bad rider guidance, and a heuristic planner can look good in a demo while being inconsistent on edge cases.

## 1. Detector accuracy

Use this when you want to measure how well YOLO detects labeled objects such as people, riders, cars, dogs, and cats.

Recommended metrics:

- `mAP50`
- `mAP50-95`
- precision
- recall
- per-class `mAP50-95`

Command:

```bash
python -m scooter_open_path_guidance_app.evaluate detect --model yolo26n.pt --data path/to/data.yaml --split val
```

Optional JSON output:

```bash
python -m scooter_open_path_guidance_app.evaluate detect --model yolo26n.pt --data path/to/data.yaml --split val --save-json runs/detect_metrics.json
```

This uses Ultralytics' validation path directly, which is the right way to benchmark YOLO on a labeled dataset.

## 2. End-to-end guidance accuracy

Use this when you want to measure whether the full system produces the correct rider command for a labeled video.

Recommended metrics:

- command accuracy
- macro precision
- macro recall
- macro F1
- confusion matrix
- optional risk-score MAE
- optional heading error in pixels

Command:

```bash
python -m scooter_open_path_guidance_app.evaluate guidance path/to/video.mp4 --annotations path/to/guidance_labels.json
```

Optional JSON output:

```bash
python -m scooter_open_path_guidance_app.evaluate guidance path/to/video.mp4 --annotations path/to/guidance_labels.json --save-json runs/guidance_metrics.json
```

## Annotation format for guidance evaluation

Create a JSON file like this:

```json
{
  "frames": [
    {
      "frame_index": 0,
      "command": "slow_down",
      "risk_score": 0.8,
      "heading_px": -40
    },
    {
      "frame_index": 15,
      "command": "maintain_speed"
    },
    {
      "frame_index": 30,
      "command": "stop",
      "risk_score": 1.0
    }
  ]
}
```

Minimum required field:

- `frame_index`
- `command`

Optional fields:

- `risk_score`
- `heading_px`

## What “good” looks like

For an early prototype, a reasonable goal is:

- detector `mAP50-95` that is stable enough on your target scene classes
- guidance command accuracy above `0.80` on a hand-labeled validation set
- strong recall on `slow_down` and `stop`

In practice, false negatives on `stop` are much more important than small drops in `speed_up` precision, so inspect the confusion matrix instead of using only one number.

## Best evaluation workflow

1. Benchmark raw detection on a public dataset or your fine-tuned dataset.
2. Build a small guidance validation set from 5 to 20 short real scooter clips.
3. Label the correct command every 10 to 15 frames, or at behavior-change points.
4. Run `evaluate guidance` after every planner change.
5. Keep the same validation clips as regression tests.
