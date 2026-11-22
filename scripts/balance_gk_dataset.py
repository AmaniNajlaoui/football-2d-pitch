#!/usr/bin/env python3
"""Oversample images that contain a goalkeeper for class-balanced fine-tuning.

Source: datasets/football-players-detection-20
Dest:   datasets/football-players-detection-20-gkbal

Copies the full dataset, then duplicates train images that have class_id=1
(goalkeeper) ``oversample`` times via hardlinks/symlinks when possible.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

GK_CLS = 1  # names: ball=0, goalkeeper=1, player=2, referee=3


def has_gk(label_path: Path) -> bool:
    text = label_path.read_text().strip()
    if not text:
        return False
    for line in text.splitlines():
        parts = line.split()
        if parts and int(parts[0]) == GK_CLS:
            return True
    return False


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.hardlink_to(src)
    except OSError:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--src",
        type=Path,
        default=Path("datasets/football-players-detection-20"),
    )
    p.add_argument(
        "--dst",
        type=Path,
        default=Path("datasets/football-players-detection-20-gkbal"),
    )
    p.add_argument("--oversample", type=int, default=10, help="Extra copies of each GK image")
    args = p.parse_args()

    src: Path = args.src
    dst: Path = args.dst
    if not (src / "data.yaml").exists():
        raise SystemExit(f"Missing {src / 'data.yaml'}")

    if dst.exists():
        shutil.rmtree(dst)

    # copy tree structure (yaml + valid/test as-is; train base copy)
    for split in ("train", "valid", "test"):
        for sub in ("images", "labels"):
            sdir = src / split / sub
            if not sdir.exists():
                continue
            for f in sdir.iterdir():
                link_or_copy(f, dst / split / sub / f.name)

    shutil.copy2(src / "data.yaml", dst / "data.yaml")
    # fix relative paths in yaml (same layout)
    yaml_text = (dst / "data.yaml").read_text()
    (dst / "data.yaml").write_text(yaml_text)

    train_labels = dst / "train" / "labels"
    gk_stems = [f.stem for f in train_labels.glob("*.txt") if has_gk(f)]
    print(f"train images with GK: {len(gk_stems)}")

    # Oversampled copies: keep only rare classes (ball/GK/ref). Dropping outfield
    # players stops the cls head from still being drowned by class 2 boxes.
    KEEP = {0, 1, 3}  # ball, goalkeeper, referee
    extra = 0
    for stem in gk_stems:
        img_candidates = list((dst / "train" / "images").glob(f"{stem}.*"))
        lab = train_labels / f"{stem}.txt"
        if not img_candidates or not lab.exists():
            continue
        img = img_candidates[0]
        kept_lines = []
        for line in lab.read_text().splitlines():
            if not line.strip():
                continue
            if int(line.split()[0]) in KEEP:
                kept_lines.append(line)
        if not any(int(l.split()[0]) == GK_CLS for l in kept_lines):
            continue
        rare_txt = "\n".join(kept_lines) + "\n"
        for k in range(args.oversample):
            new_stem = f"{stem}__gk{k}"
            link_or_copy(img, dst / "train" / "images" / f"{new_stem}{img.suffix}")
            out_lab = dst / "train" / "labels" / f"{new_stem}.txt"
            out_lab.parent.mkdir(parents=True, exist_ok=True)
            out_lab.write_text(rare_txt)
            extra += 1

    n_train = len(list((dst / "train" / "images").iterdir()))
    print(f"added {extra} GK-focused copies (no player boxes) → train images now {n_train}")
    print(f"data yaml: {dst / 'data.yaml'}")


if __name__ == "__main__":
    main()
