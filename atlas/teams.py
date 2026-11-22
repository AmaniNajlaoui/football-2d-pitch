"""Team from jersey colour in CIELAB.

White vs coloured kits need lightness L, not only (a, b).
Cluster centres are locked after a short warm-up so labels do not collapse mid-clip.
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

# Locked centres after warm-up: shape (2, 3) for (L, a, b). Team 0 = white, 1 = red.
_LOCKED_CENTRES: np.ndarray | None = None


def reset_team_model():
    """Call at the start of each new video."""
    global _LOCKED_CENTRES
    _LOCKED_CENTRES = None


def jersey_lab(frame_bgr, box, min_side=12):
    """Return mean (L, a, b) of the upper-body crop, or None."""
    x1, y1, x2, y2 = [int(v) for v in box]
    h = max(1, y2 - y1)
    w = max(1, x2 - x1)
    if h < min_side or w < min_side:
        return None
    # Shirt band: avoid head (top 10%) and legs (below 50%)
    y1b = y1 + int(0.10 * h)
    y2b = y1 + int(0.50 * h)
    # Slight inward crop to drop pitch at the box edge
    x1b = x1 + int(0.15 * w)
    x2b = x2 - int(0.15 * w)
    if x2b <= x1b + 2 or y2b <= y1b + 2:
        return None
    crop = frame_bgr[max(0, y1b):y2b, max(0, x1b):x2b]
    if crop.size == 0:
        return None
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]
    # Drop near-black (shadows) only; keep bright white kits
    keep = L > 35
    if keep.sum() < 20:
        keep = np.ones(len(lab), dtype=bool)
    return np.array([L[keep].mean(), a[keep].mean(), b[keep].mean()], np.float32)


def is_gk_kit(lab: np.ndarray | None) -> bool:
    """True if Lab shirt looks like a green/neon GK kit (not white/red outfield)."""
    if lab is None or len(lab) < 3:
        return False
    L, a, b = float(lab[0]), float(lab[1]), float(lab[2])
    # OpenCV Lab: 128 = neutral. Green → low a; lime/yellow-green → higher b.
    return a < 115 and b > 125 and 40 < L < 200


def _order_white_red(centres: np.ndarray) -> np.ndarray:
    """Return centres ordered as [white, red]. White → higher L; else higher a → red."""
    c = np.asarray(centres, np.float32).copy()
    if abs(float(c[0, 0] - c[1, 0])) >= 5:
        # brighter cluster = white
        return c if c[0, 0] >= c[1, 0] else c[::-1]
    # similar L: redder (higher a) second
    return c if c[0, 1] <= c[1, 1] else c[::-1]


def assign_teams(
    lab_by_tid: dict[int, np.ndarray],
    n_teams: int = 2,
    lock_after: int = 8,
):
    """Return tid -> 0 (white) / 1 (red) / None (referee-like).

    After ``lock_after`` tracks have features, freeze the two colour centres
    and only assign by nearest centre (stops mid-video collapse / flip).
    """
    global _LOCKED_CENTRES

    tids = [t for t, v in lab_by_tid.items() if v is not None]
    if len(tids) < 4:
        return {t: None for t in lab_by_tid}

    X = np.stack([lab_by_tid[t] for t in tids]).astype(np.float32)

    if _LOCKED_CENTRES is None:
        km = KMeans(n_clusters=n_teams, n_init=10, random_state=0)
        km.fit(X)
        centres = _order_white_red(km.cluster_centers_)
        if len(tids) >= lock_after:
            _LOCKED_CENTRES = centres
    else:
        centres = _LOCKED_CENTRES

    out = {}
    for t, lab in zip(tids, X):
        d0 = float(np.linalg.norm(lab - centres[0]))
        d1 = float(np.linalg.norm(lab - centres[1]))
        if min(d0, d1) > 45:
            out[t] = None
        else:
            out[t] = 0 if d0 <= d1 else 1
    return out
