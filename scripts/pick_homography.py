#!/usr/bin/env python3
"""Interactive src ↔ dst keypoint picker for homography (like the OpenCV example).

Left  = broadcast frame (src)
Right = 2D pitch template (dst) with known FIFA metres

Click the SAME landmark on src, then on dst. Need ≥ 4 pairs.
On dst, clicks snap to the nearest pitch landmark.

Keys (focus the matplotlib window):
  u  undo
  c  compute H
  s  save H → --out
  r  reset
  q  quit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from atlas.minimap import apply_H, draw_empty_pitch
from atlas.pitch_keypoints import PITCH_VERTICES_M, _KEYPOINT_NAMES

PAIR_COLORS = [
    "#00ff00", "#ff3333", "#33ffff", "#ff33ff",
    "#ffff33", "#ff8800", "#88ff00", "#0088ff",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", default=str(ROOT / "data" / "sample.mp4"))
    p.add_argument("--frame", type=int, default=120)
    p.add_argument("--image", default=None)
    p.add_argument("--out", default=str(ROOT / "outputs" / "H_manual.npy"))
    p.add_argument("--snap-px", type=float, default=28.0)
    return p.parse_args()


def pitch_px_to_m(px, py, map_w, map_h):
    x_m = (px - 10) / max(map_w - 20, 1) * 105.0
    y_m = (py - 10) / max(map_h - 20, 1) * 68.0
    return float(x_m), float(y_m)


def pitch_m_to_px(x_m, y_m, map_w, map_h):
    px = x_m / 105.0 * (map_w - 20) + 10
    py = y_m / 68.0 * (map_h - 20) + 10
    return float(px), float(py)


def snap_dst(px, py, map_w, map_h, snap_px):
    x_m, y_m = pitch_px_to_m(px, py, map_w, map_h)
    d = np.linalg.norm(PITCH_VERTICES_M - np.array([x_m, y_m], np.float32), axis=1)
    i = int(np.argmin(d))
    sx, sy = pitch_m_to_px(float(PITCH_VERTICES_M[i, 0]), float(PITCH_VERTICES_M[i, 1]), map_w, map_h)
    if np.hypot(sx - px, sy - py) <= snap_px:
        return sx, sy, float(PITCH_VERTICES_M[i, 0]), float(PITCH_VERTICES_M[i, 1]), _KEYPOINT_NAMES[i]
    return float(px), float(py), x_m, y_m, None


def load_frame(args):
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            sys.exit(f"Cannot read image: {args.image}")
        return frame, args.frame
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {args.video}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit(f"Cannot read frame {args.frame}")
    return frame, args.frame


def main():
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    frame, frame_idx = load_frame(args)
    src_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w = frame.shape[:2]
    dst_bgr, _ = draw_empty_pitch(h=h, w=int(h * 105 / 68))
    map_h, map_w = dst_bgr.shape[:2]
    dst_rgb = cv2.cvtColor(dst_bgr, cv2.COLOR_BGR2RGB)

    state = {
        "src_pts": [],
        "dst_pts_m": [],
        "dst_pts_px": [],
        "names": [],
        "expect": "src",
        "H": None,
        "status": "Click SRC landmark, then same point on DST (≥4 pairs). Keys: u c s r q",
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.canvas.manager.set_window_title("Homography keypoint picker  ·  src | dst")
    ax_src, ax_dst = axes
    im_src = ax_src.imshow(src_rgb)
    im_dst = ax_dst.imshow(dst_rgb)
    ax_src.set_title("src (broadcast)")
    ax_dst.set_title("dst (2D pitch, metres)")
    ax_src.axis("off")
    ax_dst.axis("off")

    # guide landmarks on dst
    for xm, ym in PITCH_VERTICES_M:
        px, py = pitch_m_to_px(float(xm), float(ym), map_w, map_h)
        ax_dst.plot(px, py, "o", color="#2ecc71", markersize=4, alpha=0.7)

    src_scats, dst_scats, src_texts, dst_texts = [], [], [], []
    status_txt = fig.text(0.5, 0.02, state["status"], ha="center", fontsize=11, color="black")

    def redraw_points():
        for artist in src_scats + dst_scats + src_texts + dst_texts:
            artist.remove()
        src_scats.clear()
        dst_scats.clear()
        src_texts.clear()
        dst_texts.clear()
        n = len(state["dst_pts_m"])
        for i in range(n):
            color = PAIR_COLORS[i % len(PAIR_COLORS)]
            sx, sy = state["src_pts"][i]
            dx, dy = state["dst_pts_px"][i]
            src_scats.append(ax_src.plot(sx, sy, "o", color=color, markersize=12)[0])
            dst_scats.append(ax_dst.plot(dx, dy, "o", color=color, markersize=12)[0])
            src_texts.append(ax_src.text(sx + 8, sy - 8, str(i + 1), color=color, fontsize=12, fontweight="bold"))
            dst_texts.append(ax_dst.text(dx + 8, dy - 8, str(i + 1), color=color, fontsize=12, fontweight="bold"))
        if len(state["src_pts"]) > n:
            i = n
            color = PAIR_COLORS[i % len(PAIR_COLORS)]
            sx, sy = state["src_pts"][i]
            src_scats.append(ax_src.plot(sx, sy, "o", color=color, markersize=12)[0])
            src_texts.append(ax_src.text(sx + 8, sy - 8, f"{i+1}?", color=color, fontsize=12, fontweight="bold"))
        status_txt.set_text(state["status"])
        fig.canvas.draw_idle()

    def compute():
        n = min(len(state["src_pts"]), len(state["dst_pts_m"]))
        if n < 4:
            state["status"] = f"Need ≥4 pairs (have {n})."
            redraw_points()
            return False
        src = np.float32(state["src_pts"][:n])
        dst = np.float32(state["dst_pts_m"][:n])
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        H, mask = cv2.findHomography(src, dst, method, 3.0)
        if H is None:
            state["status"] = "findHomography failed."
            redraw_points()
            return False
        state["H"] = H
        nin = int(mask.sum()) if mask is not None else n
        err = np.linalg.norm(apply_H(H, src) - dst, axis=1)
        state["status"] = f"H OK — {nin}/{n} inliers, median err {float(np.median(err)):.2f} m  (press s to save)"
        print(state["status"])
        for i in range(n):
            print(f"  {i+1}. src={src[i]} → {state['names'][i]} {dst[i]}  err={err[i]:.2f}m")
        redraw_points()
        return True

    def on_click(event):
        if event.inaxes is None or event.xdata is None:
            return
        x, y = float(event.xdata), float(event.ydata)
        if event.inaxes is ax_src:
            if state["expect"] != "src":
                state["status"] = "Click DST next (same landmark)."
            else:
                state["src_pts"].append((x, y))
                state["expect"] = "dst"
                state["status"] = f"SRC #{len(state['src_pts'])} set → click same landmark on DST"
        elif event.inaxes is ax_dst:
            if state["expect"] != "dst":
                state["status"] = "Click SRC first."
            else:
                sx, sy, xm, ym, name = snap_dst(x, y, map_w, map_h, args.snap_px)
                state["dst_pts_px"].append((sx, sy))
                state["dst_pts_m"].append((xm, ym))
                state["names"].append(name or f"({xm:.1f},{ym:.1f})")
                state["expect"] = "src"
                n = len(state["dst_pts_m"])
                label = name or f"{xm:.1f}m,{ym:.1f}m"
                state["status"] = f"Pair {n}: {label}" + (
                    " — press c to compute H" if n >= 4 else f" — need {4 - n} more"
                )
        redraw_points()

    def on_key(event):
        key = (event.key or "").lower()
        if key == "q" or key == "escape":
            plt.close(fig)
        elif key == "u":
            if state["expect"] == "dst" and state["src_pts"]:
                state["src_pts"].pop()
                state["expect"] = "src"
                state["status"] = "Undid SRC click."
            elif state["src_pts"] and state["dst_pts_m"]:
                state["src_pts"].pop()
                state["dst_pts_m"].pop()
                state["dst_pts_px"].pop()
                state["names"].pop()
                state["status"] = "Undid last pair."
            state["H"] = None
            redraw_points()
        elif key == "r":
            state.update(src_pts=[], dst_pts_m=[], dst_pts_px=[], names=[], expect="src", H=None,
                         status="Reset. Click SRC then DST.")
            redraw_points()
        elif key == "c":
            compute()
        elif key == "s":
            if state["H"] is None and not compute():
                return
            np.save(out, state["H"])
            meta = {
                "src": state["src_pts"],
                "dst_m": state["dst_pts_m"],
                "names": state["names"],
                "frame": frame_idx,
            }
            meta_path = out.with_suffix(".json")
            meta_path.write_text(json.dumps(meta, indent=2))
            state["status"] = f"Saved {out.name} + {meta_path.name}"
            print(state["status"])
            redraw_points()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    print(__doc__)
    print(f"Frame {frame_idx}. Good picks: halfway↔touchline, centre-circle intersections, pen-box corners.")
    plt.show()


if __name__ == "__main__":
    main()
