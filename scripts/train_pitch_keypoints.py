#!/usr/bin/env python3
"""Fine-tune YOLO-pose pitch keypoint detector (Roboflow football-field-detection).

Follows Roboflow sports tutorial:
  mosaic=0.0  (mosaic hurts pitch geometry)
  32 keypoints / image

NOTE: yolov8x-pose + batch 48 needs an A100. On a 4GB laptop GPU we use
yolov8n-pose (or s) with a small batch. Override with --model / --batch.

Usage:
  export ROBOFLOW_API_KEY=...   # only needed if dataset missing
  python scripts/train_pitch_keypoints.py
  python scripts/train_pitch_keypoints.py --model yolov8s-pose.pt --epochs 100 --batch 2
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "datasets" / "football-field-detection-18"
DATA_YAML = DATA_DIR / "data.yaml"


def ensure_dataset(api_key: str | None):
    if DATA_YAML.exists():
        return
    if not api_key:
        sys.exit(
            f"Dataset missing at {DATA_DIR}. Set ROBOFLOW_API_KEY or download v18 first."
        )
    from roboflow import Roboflow

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("roboflow-jvuqo").project("football-field-detection-f07vi")
    project.version(18).download("yolov8", location=str(DATA_DIR))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default="yolov8n-pose.pt",
        help="Start weights. Use yolov8x-pose.pt only if you have ≥16GB VRAM.",
    )
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--name", default="pitch_keypoints")
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dataset(os.environ.get("ROBOFLOW_API_KEY"))
    if not DATA_YAML.exists():
        sys.exit(f"No data.yaml at {DATA_YAML}")

    from ultralytics import YOLO

    print(f"Train {args.model} on {DATA_YAML}")
    print("mosaic=0.0 (Roboflow recommendation for pitch keypoints)")
    model = YOLO(args.model)
    model.train(
        data=str(DATA_YAML),
        task="pose",
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        mosaic=0.0,
        device=args.device,
        project=str(ROOT / "runs" / "pose"),
        name=args.name,
        exist_ok=True,
        patience=20,
        plots=True,
    )
    best = ROOT / "runs" / "pose" / args.name / "weights" / "best.pt"
    print(f"Best weights → {best}")
    print("Point atlas/pitch_keypoints.py WEIGHTS at this file (or copy to weights/).")


if __name__ == "__main__":
    main()
