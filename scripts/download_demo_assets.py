#!/usr/bin/env python3
"""Download a short football clip. Detector is official YOLO26 (auto-downloaded)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "data").mkdir(exist_ok=True)
out = ROOT / "data" / "sample.mp4"

# Roboflow sports broadcast clips (DFL / soccer examples).
CLIPS = [
    ("12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF", "0bfacc_0.mp4"),
    ("1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-", "08fd33_0.mp4"),
    ("19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf", "2e57b9_0.mp4"),
]


def looks_like_clip(path: Path) -> bool:
    return path.exists() and path.stat().st_size > 3_000_000


def main():
    if looks_like_clip(out) and out.stat().st_size != 5_800_320:
        print("Already have a clip:", out, out.stat().st_size)
        return

    import gdown

    for file_id, name in CLIPS:
        print(f"Downloading {name} …")
        try:
            gdown.download(id=file_id, output=str(out), quiet=False)
        except Exception as e:
            print("  failed:", e)
            continue
        if looks_like_clip(out):
            print("OK:", out, out.stat().st_size)
            return

    raise SystemExit(f"Could not download a football clip. Put one at {out}")


if __name__ == "__main__":
    main()
