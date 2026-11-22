#!/usr/bin/env python3
"""Fine-tune player detector with GK-balanced data so goalkeeper ≠ player.

Starts from runs/detect/train/weights/best.pt and trains on the oversampled
GK dataset (scripts/balance_gk_dataset.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        default=str(ROOT / "runs" / "detect" / "train" / "weights" / "best.pt"),
    )
    p.add_argument(
        "--data",
        default=str(ROOT / "datasets" / "football-players-detection-20-gkbal" / "data.yaml"),
    )
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--name", default="train_gk")
    args = p.parse_args()

    data = Path(args.data)
    if not data.exists():
        raise SystemExit(
            f"Missing {data}. Run: python scripts/balance_gk_dataset.py"
        )

    from ultralytics import YOLO

    model = YOLO(args.model)
    # Stronger classification signal + copy-paste helps rare GK boxes.
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        exist_ok=True,
        pretrained=True,
        patience=20,
        # emphasize class head (default cls=0.5)
        cls=1.5,
        copy_paste=0.5,
        mosaic=1.0,
        close_mosaic=10,
        hsv_h=0.02,
        hsv_s=0.7,
        hsv_v=0.4,
        fliplr=0.5,
        workers=8,
        plots=True,
    )
    best = ROOT / "runs" / "detect" / args.name / "weights" / "best.pt"
    print(f"Done. Best weights: {best}")


if __name__ == "__main__":
    main()
