"""Ball detection helpers + temporal coasting for continuous 2D path."""
from __future__ import annotations

from collections import deque

import cv2
import numpy as np


class BallTracker:
    """Keep a continuous ball position on the pitch even when YOLO misses frames.

    - Accepts image detections (centre of bbox)
    - Projects with H → metres
    - EMA smooth + coast (hold / light velocity) while missing
    - Stores a trail for the minimap
    """

    def __init__(
        self,
        conf_thr: float = 0.05,
        alpha: float = 0.55,
        max_coast_frames: int = 12,
        max_jump_m: float = 4.5,
        trail_len: int = 400,
    ):
        self.conf_thr = float(conf_thr)
        self.alpha = float(alpha)
        self.max_coast = int(max_coast_frames)
        self.max_jump_m = float(max_jump_m)
        self.trail: deque[tuple[float, float]] = deque(maxlen=int(trail_len))
        self.pos: np.ndarray | None = None
        self.vel = np.zeros(2, np.float64)
        self.missed = 0
        self.last_img_xy: np.ndarray | None = None

    def _grass_frac(self, frame_bgr, cx: float, cy: float, bw: float, bh: float) -> float:
        """Share of green around the box. A real ball sits on the pitch; logos and crowd do not."""
        h, w = frame_bgr.shape[:2]
        r = int(max(10.0, max(bw, bh) * 2.4))
        x1, y1 = max(0, int(cx - r)), max(0, int(cy - r))
        x2, y2 = min(w, int(cx + r)), min(h, int(cy + r))
        patch = frame_bgr[y1:y2, x1:x2]
        if patch.size < 30:
            return 0.0
        hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        green = (
            (hsv[:, :, 0] >= 28)
            & (hsv[:, :, 0] <= 95)
            & (hsv[:, :, 1] >= 30)
            & (hsv[:, :, 2] >= 30)
        )
        return float(green.mean())

    def detect_image(self, model, frame_bgr, imgsz: int = 1280) -> tuple[np.ndarray, float] | None:
        """Return ball centre (x,y) in image pixels + conf, or None.

        Keeps only small boxes on grass. After the track has started, a new box
        must stay near the previous image position — the highest-confidence box
        is often a shirt, boot, or logo.
        """
        r = model.predict(
            frame_bgr,
            conf=self.conf_thr,
            imgsz=imgsz,
            classes=[0],  # ball
            verbose=False,
        )[0]
        if r.boxes is None or len(r.boxes) == 0:
            return None
        xyxy = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        h, w = frame_bgr.shape[:2]
        cands = []
        for box, cf in zip(xyxy, conf):
            x1, y1, x2, y2 = box
            bw, bh = float(x2 - x1), float(y2 - y1)
            if bw < 2 or bh < 2:
                continue
            # Broadcast ball is a few dozen pixels, not a person-sized box.
            if bw > 48 or bh > 48 or bw > 0.035 * w or bh > 0.05 * h:
                continue
            if max(bw, bh) / max(min(bw, bh), 1.0) > 2.2:
                continue
            cx, cy = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
            if self._grass_frac(frame_bgr, cx, cy, bw, bh) < 0.28:
                continue
            cands.append((float(cx), float(cy), float(cf)))
        if not cands:
            return None
        if self.last_img_xy is not None:
            gap = max(1, self.missed + 1)
            limit = min(260.0, 80.0 * gap)
            lx, ly = float(self.last_img_xy[0]), float(self.last_img_xy[1])
            near = [c for c in cands if np.hypot(c[0] - lx, c[1] - ly) <= limit]
            if not near:
                return None
            near.sort(key=lambda c: (np.hypot(c[0] - lx, c[1] - ly), -c[2]))
            cx, cy, cf = near[0]
        else:
            strong = [c for c in cands if c[2] >= 0.22]
            if not strong:
                return None
            strong.sort(key=lambda c: -c[2])
            cx, cy, cf = strong[0]
        return np.array([cx, cy], np.float64), cf

    def update(self, xy_m: np.ndarray | None) -> np.ndarray | None:
        """Update from pitch metres (or None if no detection this frame)."""
        if xy_m is not None:
            xy = np.asarray(xy_m, np.float64).reshape(2)
            on_pitch = (-2.0 <= xy[0] <= 107.0) and (-2.0 <= xy[1] <= 70.0)
            if not on_pitch:
                xy_m = None
            else:
                if self.pos is None:
                    self.pos = xy.copy()
                    self.vel[:] = 0
                else:
                    jump = float(np.linalg.norm(xy - self.pos))
                    # Allow a longer step only while the detector was missing.
                    limit = min(16.0, self.max_jump_m * max(1, self.missed + 1))
                    if jump > limit:
                        xy_m = None
                    else:
                        new_pos = self.alpha * xy + (1.0 - self.alpha) * self.pos
                        self.vel = new_pos - self.pos
                        self.pos = new_pos
                if xy_m is not None:
                    self.missed = 0
                    self.trail.append((float(self.pos[0]), float(self.pos[1])))
                    return self.pos.copy()

        # no trustworthy detection — hold briefly, do not extend the drawn trail
        self.missed += 1
        if self.pos is None or self.missed > self.max_coast:
            if self.missed > self.max_coast:
                self.pos = None
                self.last_img_xy = None
                self.vel[:] = 0
            return None
        return self.pos.copy()


def draw_ball_trail(pitch_img, xy_fn, trail, color=(0, 220, 255)):
    """Draw fading path of ball positions on the minimap."""
    if not trail or len(trail) < 2:
        return pitch_img
    vis = pitch_img
    h, w = vis.shape[:2]
    pts = []
    for xm, ym in trail:
        p = xy_fn(float(np.clip(xm, 0, 105)), float(np.clip(ym, 0, 68)))
        px = int(np.clip(p[0], 0, w - 1))
        py = int(np.clip(p[1], 0, h - 1))
        pts.append((px, py))
    n = len(pts)
    # Break the line when two accepted positions are too far apart (false jump).
    def _far(a, b) -> bool:
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 > 55 ** 2

    for i in range(1, n):
        if _far(pts[i - 1], pts[i]):
            continue
        cv2.line(vis, pts[i - 1], pts[i], (20, 20, 20), 4, cv2.LINE_AA)
    for i in range(1, n):
        if _far(pts[i - 1], pts[i]):
            continue
        t = i / max(n - 1, 1)
        thickness = 2 if t < 0.85 else 3
        c = (
            int(color[0] * (0.45 + 0.55 * t)),
            int(color[1] * (0.45 + 0.55 * t)),
            int(color[2] * (0.45 + 0.55 * t)),
        )
        cv2.line(vis, pts[i - 1], pts[i], c, thickness, cv2.LINE_AA)
    cv2.circle(vis, pts[-1], 11, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.circle(vis, pts[-1], 7, color, -1)
    cv2.circle(vis, pts[-1], 7, (20, 20, 20), 1, cv2.LINE_AA)
    return vis
