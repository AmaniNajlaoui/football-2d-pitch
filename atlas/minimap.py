"""Draw a 2D FIFA pitch and plot tracked people / ball."""
from __future__ import annotations

import cv2
import numpy as np

PITCH_L, PITCH_W = 105.0, 68.0


def _line(img, p1, p2, color=(255, 255, 255), t=1):
    cv2.line(img, p1, p2, color, t, cv2.LINE_AA)


def draw_empty_pitch(h=680, w=1050):
    img = np.full((h, w, 3), (46, 120, 62), np.uint8)

    def xy(x_m, y_m):
        # pitch coords: x in [0,105] length, y in [0,68] width
        px = int(x_m / PITCH_L * (w - 20) + 10)
        py = int(y_m / PITCH_W * (h - 20) + 10)
        return px, py

    _line(img, xy(0, 0), xy(PITCH_L, 0))
    _line(img, xy(PITCH_L, 0), xy(PITCH_L, PITCH_W))
    _line(img, xy(PITCH_L, PITCH_W), xy(0, PITCH_W))
    _line(img, xy(0, PITCH_W), xy(0, 0))
    _line(img, xy(PITCH_L / 2, 0), xy(PITCH_L / 2, PITCH_W))
    cv2.circle(img, xy(PITCH_L / 2, PITCH_W / 2), int(9.15 / PITCH_W * (h - 20)), (255, 255, 255), 1)
    for x0 in (0.0, PITCH_L - 16.5):
        _line(img, xy(x0, (PITCH_W - 40.32) / 2), xy(x0 + 16.5, (PITCH_W - 40.32) / 2))
        _line(img, xy(x0 + 16.5, (PITCH_W - 40.32) / 2), xy(x0 + 16.5, (PITCH_W + 40.32) / 2))
        _line(img, xy(x0 + 16.5, (PITCH_W + 40.32) / 2), xy(x0, (PITCH_W + 40.32) / 2))
    for x0 in (0.0, PITCH_L - 5.5):
        y0 = (PITCH_W - 18.32) / 2
        _line(img, xy(x0, y0), xy(x0 + 5.5, y0))
        _line(img, xy(x0 + 5.5, y0), xy(x0 + 5.5, y0 + 18.32))
        _line(img, xy(x0 + 5.5, y0 + 18.32), xy(x0, y0 + 18.32))
    return img, xy


def apply_H(H, pts):
    """pts: Nx2 image pixels → Nx2 pitch metres (x,y)."""
    if H is None or len(pts) == 0:
        return np.zeros((0, 2), np.float32)
    p = np.concatenate([pts.astype(np.float64), np.ones((len(pts), 1), np.float64)], 1)
    q = (np.asarray(H, np.float64) @ p.T).T
    # Homogeneous w can be negative — only guard near-zero (do NOT clip to +eps).
    w = q[:, 2:3]
    w = np.where(np.abs(w) < 1e-8, np.sign(w) * 1e-8 + (w == 0) * 1e-8, w)
    return (q[:, :2] / w).astype(np.float32)


def stamp_objects(pitch_img, xy, objects):
    """objects: list of dicts with keys x_m, y_m, kind, team, tid."""
    vis = pitch_img.copy()
    h, w = vis.shape[:2]
    margin = 14  # keep markers/labels inside the canvas
    for o in objects:
        if o.get("x_m") is None:
            continue
        # Sideline ARs often land at y≈68–72 m; clamp onto the pitch so they
        # still draw (previously skipped when pixel y >= map height).
        xm = float(np.clip(o["x_m"], 0.0, PITCH_L))
        ym = float(np.clip(o["y_m"], 0.0, PITCH_W))
        p = xy(xm, ym)
        px = int(np.clip(p[0], margin, w - margin - 1))
        py = int(np.clip(p[1], margin, h - margin - 1))
        p = (px, py)
        kind = o.get("kind", "player")
        tid = str(o.get("tid", ""))
        if kind == "ball":
            cv2.circle(vis, p, 6, (0, 220, 255), -1)  # yellow ball
            cv2.circle(vis, p, 6, (20, 20, 20), 1)
        elif kind == "referee":
            cv2.drawMarker(vis, p, (0, 215, 255), cv2.MARKER_TILTED_CROSS, 16, 2)
            cv2.putText(vis, tid, (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(vis, tid, (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1, cv2.LINE_AA)
        elif kind == "goalkeeper":
            # Green square — distinct from outfield dots
            cv2.drawMarker(vis, p, (0, 200, 0), cv2.MARKER_SQUARE, 14, 2)
            cv2.putText(vis, f"GK{tid}", (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (20, 20, 20), 2, cv2.LINE_AA)
            cv2.putText(vis, f"GK{tid}", (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 0), 1, cv2.LINE_AA)
        else:
            # team 0 white, team 1 red
            if o.get("team") == 0:
                color = (255, 255, 255)
            elif o.get("team") == 1:
                color = (0, 0, 255)
            else:
                color = (200, 200, 200)
            cv2.circle(vis, p, 8, color, -1)
            cv2.circle(vis, p, 8, (20, 20, 20), 1)
            cv2.putText(vis, tid, (p[0] + 6, p[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    return vis
