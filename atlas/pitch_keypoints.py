"""Homography from real pitch landmarks (known FIFA measurements).

Uses Roboflow yolov8x pitch keypoint model (32 points). For a moving camera we
re-detect every frame. Prefer centre-circle + halfway landmarks (most stable on
broadcast views); reject bad H updates.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights" / "football-pitch-detection.pt"
GDRIVE_ID = "1Ma5Kt86tgpdjCTKfum79YMgNnSjcoOyf"

PITCH_L, PITCH_W = 105.0, 68.0
# FIFA metres (do NOT anisotropically scale a 120×70 diagram — that shrinks the
# centre circle to ~8.0×8.9 m and stretches players near the touchlines).
_PEN_D, _PEN_W = 16.5, 40.32
_GOAL_D, _GOAL_W = 5.5, 18.32
_SPOT, _CIRCLE_R = 11.0, 9.15
_CY = PITCH_W / 2.0


PITCH_VERTICES_M = np.array(
    [
        (0.0, 0.0),
        (0.0, (PITCH_W - _PEN_W) / 2),
        (0.0, (PITCH_W - _GOAL_W) / 2),
        (0.0, (PITCH_W + _GOAL_W) / 2),
        (0.0, (PITCH_W + _PEN_W) / 2),
        (0.0, PITCH_W),
        (_GOAL_D, (PITCH_W - _GOAL_W) / 2),
        (_GOAL_D, (PITCH_W + _GOAL_W) / 2),
        (_SPOT, _CY),
        (_PEN_D, (PITCH_W - _PEN_W) / 2),
        (_PEN_D, (PITCH_W - _GOAL_W) / 2),
        (_PEN_D, (PITCH_W + _GOAL_W) / 2),
        (_PEN_D, (PITCH_W + _PEN_W) / 2),
        (PITCH_L / 2, 0.0),
        (PITCH_L / 2, _CY - _CIRCLE_R),
        (PITCH_L / 2, _CY + _CIRCLE_R),
        (PITCH_L / 2, PITCH_W),
        (PITCH_L - _PEN_D, (PITCH_W - _PEN_W) / 2),
        (PITCH_L - _PEN_D, (PITCH_W - _GOAL_W) / 2),
        (PITCH_L - _PEN_D, (PITCH_W + _GOAL_W) / 2),
        (PITCH_L - _PEN_D, (PITCH_W + _PEN_W) / 2),
        (PITCH_L - _SPOT, _CY),
        (PITCH_L - _GOAL_D, (PITCH_W - _GOAL_W) / 2),
        (PITCH_L - _GOAL_D, (PITCH_W + _GOAL_W) / 2),
        (PITCH_L, 0.0),
        (PITCH_L, (PITCH_W - _PEN_W) / 2),
        (PITCH_L, (PITCH_W - _GOAL_W) / 2),
        (PITCH_L, (PITCH_W + _GOAL_W) / 2),
        (PITCH_L, (PITCH_W + _PEN_W) / 2),
        (PITCH_L, PITCH_W),
        (PITCH_L / 2 - _CIRCLE_R, _CY),
        (PITCH_L / 2 + _CIRCLE_R, _CY),
    ],
    dtype=np.float32,
)

_KEYPOINT_NAMES = [
    "TL corner", "L pen box top", "L goal box top", "L goal box bottom", "L pen box bottom", "BL corner",
    "L goal area top", "L goal area bottom", "L penalty spot",
    "L pen area top", "L pen/goal top", "L pen/goal bottom", "L pen area bottom",
    "Halfway TOP", "Centre circle N", "Centre circle S", "Halfway BOTTOM",
    "R pen area top", "R pen/goal top", "R pen/goal bottom", "R pen area bottom",
    "R penalty spot", "R goal area top", "R goal area bottom",
    "TR corner", "R pen box top", "R goal box top", "R goal box bottom", "R pen box bottom", "BR corner",
    "Centre circle W", "Centre circle E",
]

_CORNER_IDX = {0, 5, 24, 29}
# Most reliable on typical broadcast (centre + halfway). Prefer these for H.
_CENTRE_IDX = {13, 14, 15, 16, 30, 31}
# Circle N/S/W/E — model often sits slightly outside the paint
_CIRCLE_IDX = {14, 15, 30, 31}
_HALFWAY_IDX = {13, 16}
_TRUSTED_IDX = _CENTRE_IDX | {8, 9, 10, 11, 12, 17, 18, 19, 20, 21}


def _line_ridge_score(bgr: np.ndarray) -> np.ndarray:
    """Local brightness ridge — works on worn / grey pitch paint."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    L = lab[:, :, 0].astype(np.float32)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    # keep roughly green / neutral field; drop spectators / ads
    chroma = np.hypot(a - 128.0, b - 128.0)
    greenish = ((a < 140) & (b > 110) & (L > 40) & (L < 230)).astype(np.float32)
    local = cv2.GaussianBlur(L, (31, 31), 0)
    ridge = np.clip(L - local, 0.0, 40.0) / 40.0
    # slight absolute-white bonus when paint is clean
    white = np.clip((L - 150.0) / 80.0, 0.0, 1.0) * np.clip(1.0 - chroma / 40.0, 0.0, 1.0)
    score = (0.75 * ridge + 0.25 * white) * greenish
    return cv2.GaussianBlur(score, (3, 3), 0)


def _sample_score(score: np.ndarray, x: float, y: float) -> float:
    h, w = score.shape
    if x < 0 or y < 0 or x >= w - 1 or y >= h - 1:
        return -1.0
    x0, y0 = int(x), int(y)
    dx, dy = x - x0, y - y0
    s00 = score[y0, x0]
    s10 = score[y0, x0 + 1]
    s01 = score[y0 + 1, x0]
    s11 = score[y0 + 1, x0 + 1]
    return float((1 - dx) * (1 - dy) * s00 + dx * (1 - dy) * s10 + (1 - dx) * dy * s01 + dx * dy * s11)


def _best_along_ray(
    score: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
    r_min: float,
    r_max: float,
    step: float = 0.5,
    min_score: float = 0.08,
    prefer_outer: bool = False,
) -> tuple[np.ndarray, float] | None:
    """Find ridge along a ray. If prefer_outer, take the outermost local peak
    (avoids locking onto the halfway line near the circle centre)."""
    n = float(np.linalg.norm(direction))
    if n < 1e-6:
        return None
    u = direction / n
    rs, ss = [], []
    r = r_min
    while r <= r_max + 1e-6:
        p = origin + u * r
        rs.append(r)
        ss.append(_sample_score(score, float(p[0]), float(p[1])))
        r += step
    if not ss:
        return None
    ss = np.asarray(ss, np.float64)
    rs = np.asarray(rs, np.float64)

    if prefer_outer:
        # walk inward from r_max; first sustained peak = circle edge
        thr = max(min_score, 0.45 * float(ss.max()) if ss.max() > 0 else min_score)
        i = len(ss) - 1
        while i >= 1 and ss[i] < thr:
            i -= 1
        if ss[i] < min_score:
            return None
        # refine to local max near this outer hit
        j0 = max(0, i - 4)
        j1 = min(len(ss), i + 5)
        j = j0 + int(np.argmax(ss[j0:j1]))
        return origin + u * float(rs[j]), float(ss[j])

    j = int(np.argmax(ss))
    if ss[j] < min_score:
        return None
    return origin + u * float(rs[j]), float(ss[j])


def _local_peak(
    score: np.ndarray,
    p: np.ndarray,
    rad: int = 16,
    min_score: float = 0.08,
) -> np.ndarray | None:
    best, best_s = None, -1.0
    for dy in range(-rad, rad + 1):
        for dx in range(-rad, rad + 1):
            if dx * dx + dy * dy > rad * rad:
                continue
            x, y = float(p[0] + dx), float(p[1] + dy)
            s = _sample_score(score, x, y)
            if s > best_s:
                best_s = s
                best = np.array([x, y], dtype=np.float64)
    if best is None or best_s < min_score:
        return None
    return best


def _walk_inward_first_ridge(
    score: np.ndarray,
    p: np.ndarray,
    center: np.ndarray,
    max_dist: float,
    step: float = 0.5,
    min_score: float = 0.05,
    lateral: float = 0.0,
    tang: np.ndarray | None = None,
) -> np.ndarray | None:
    """From a point outside the paint, walk toward centre; stop at first ridge peak."""
    v = center - p
    n = float(np.linalg.norm(v))
    if n < 1e-6:
        return None
    u = v / n
    origin = p.copy()
    if tang is not None and abs(lateral) > 1e-6:
        origin = origin + tang * lateral
    # scores along the inward walk
    rs, ss = [], []
    r = 0.0
    while r <= max_dist + 1e-6:
        q = origin + u * r
        rs.append(r)
        ss.append(_sample_score(score, float(q[0]), float(q[1])))
        r += step
    ss = np.asarray(ss, np.float64)
    if ss.size < 5 or float(ss.max()) < min_score:
        return None
    thr = max(min_score, 0.50 * float(ss.max()))
    # first contiguous blob above thr → sit just inside the outer paint edge
    # (peak of thick worn lines often drifts inward onto grass)
    above = ss >= thr
    if not np.any(above):
        return None
    i0 = int(np.argmax(above))  # first True = outer edge
    i1 = i0
    while i1 + 1 < len(ss) and above[i1 + 1]:
        i1 += 1
    # 1–2 samples into the stripe ≈ centre of a ~3–5px line
    j = min(i1, i0 + max(1, (i1 - i0) // 3))
    # reject if we walked almost to centre (hit halfway line instead)
    if rs[j] > 0.90 * n:
        return None
    return origin + u * float(rs[j])


def snap_keypoints_to_paint(bgr: np.ndarray, xy: np.ndarray, conf: np.ndarray) -> np.ndarray:
    """Pull YOLO landmarks onto pitch-line ridges (fixes outward centre-circle bias)."""
    out = xy.astype(np.float64).copy()
    score = _line_ridge_score(bgr)

    # Halfway TOP/BOTTOM first (defines the tilted halfway line)
    for i in _HALFWAY_IDX:
        if i >= len(out) or out[i, 0] <= 1 or conf[i] < 0.5:
            continue
        p = out[i]
        best, best_s = p.copy(), -1.0
        for dx in range(-18, 19):
            for dy in range(-12, 13):
                x, y = p[0] + dx, p[1] + dy
                s = _sample_score(score, x, y)
                s -= 0.0015 * abs(dy)
                if s > best_s:
                    best_s = s
                    best = np.array([x, y], dtype=np.float64)
        if best_s >= 0.05:
            out[i] = best

    ns = [out[i] for i in (14, 15) if out[i, 0] > 1 and conf[i] >= 0.5]
    we = [out[i] for i in (30, 31) if out[i, 0] > 1 and conf[i] >= 0.5]
    if len(ns) == 2:
        center = 0.5 * (ns[0] + ns[1])
    elif len(we) == 2:
        center = 0.5 * (we[0] + we[1])
    else:
        pts = ns + we
        if len(pts) < 2:
            return out
        center = np.mean(np.stack(pts, 0), axis=0)

    # Halfway line from TOP→BOTTOM (tilted under broadcast perspective)
    hw_a = hw_b = None
    if out[13, 0] > 1 and conf[13] >= 0.5 and out[16, 0] > 1 and conf[16] >= 0.5:
        hw_a, hw_b = out[13].copy(), out[16].copy()
    elif len(ns) == 2:
        hw_a, hw_b = ns[0].copy(), ns[1].copy()

    def _proj_halfway(p: np.ndarray) -> np.ndarray:
        if hw_a is None or hw_b is None:
            return p
        ab = hw_b - hw_a
        denom = float(np.dot(ab, ab))
        if denom < 1e-6:
            return p
        t = float(np.dot(p - hw_a, ab) / denom)
        return hw_a + t * ab

    # N/S: walk inward to first ridge, then project onto halfway line
    for i in (14, 15):
        if i >= len(out) or out[i, 0] <= 1 or conf[i] < 0.5:
            continue
        p = out[i]
        dist = float(np.linalg.norm(p - center))
        hit = _walk_inward_first_ridge(score, p, center, max_dist=min(28.0, 0.35 * dist))
        if hit is not None:
            out[i] = hit
        else:
            peak = _local_peak(score, p, rad=12, min_score=0.05)
            if peak is not None and np.linalg.norm(peak - center) <= dist + 0.5:
                out[i] = peak
        # pin to halfway, then micro-search along the line for paint
        base = _proj_halfway(out[i])
        if hw_a is not None:
            ab = hw_b - hw_a
            abn = ab / (np.linalg.norm(ab) + 1e-8)
            best, best_s = base.copy(), _sample_score(score, float(base[0]), float(base[1]))
            for t in range(-10, 11):
                q = base + abn * t
                s = _sample_score(score, float(q[0]), float(q[1]))
                if s > best_s:
                    best_s = s
                    best = q
            if best_s >= 0.04:
                out[i] = best
            else:
                out[i] = base

    if out[14, 0] > 1 and out[15, 0] > 1:
        center = 0.5 * (out[14] + out[15])

    # W/E: walk inward from YOLO (outside) → first ridge = outer paint edge
    for i in (30, 31):
        if i >= len(out) or out[i, 0] <= 1 or conf[i] < 0.5:
            continue
        p = out[i]
        dist = float(np.linalg.norm(p - center))
        if dist < 5:
            continue
        # don't walk past a plausible ellipse radius from N/S
        max_walk = min(dist - 8.0, 0.55 * dist)
        if out[14, 0] > 1 and out[15, 0] > 1:
            r_ns = 0.5 * (
                float(np.linalg.norm(out[14] - center))
                + float(np.linalg.norm(out[15] - center))
            )
            # leave at least ~1.15 * r_ns from centre (ellipse floor)
            max_walk = min(max_walk, dist - 1.15 * r_ns)
        if max_walk < 3:
            continue
        tang = np.array([-(p - center)[1], (p - center)[0]], dtype=np.float64)
        tn = float(np.linalg.norm(tang))
        tang = tang / tn if tn > 1e-6 else np.array([0.0, 1.0])
        best_pt, best_inward = None, -1.0
        for lat in (-6.0, -3.0, 0.0, 3.0, 6.0):
            hit = _walk_inward_first_ridge(
                score, p, center, max_dist=max_walk, lateral=lat, tang=tang
            )
            if hit is None:
                continue
            inward = float(np.linalg.norm(p - hit))
            s = _sample_score(score, float(hit[0]), float(hit[1]))
            rank = s + 0.002 * inward
            if rank > best_inward:
                best_inward = rank
                best_pt = hit
        if best_pt is not None:
            out[i] = best_pt

    for i in _TRUSTED_IDX - _CIRCLE_IDX - _HALFWAY_IDX:
        if i >= len(out) or out[i, 0] <= 1 or conf[i] < 0.7:
            continue
        peak = _local_peak(score, out[i], rad=10, min_score=0.08)
        if peak is not None:
            out[i] = peak
    return out


class PitchKeypointCalibrator:
    """Detect pitch landmarks each frame → H (image → metres)."""

    def __init__(
        self,
        weights: str | Path | None = None,
        conf_thr: float = 0.7,
        detect_every: int = 1,
        h_alpha: float = 0.5,
        snap_to_paint: bool = False,
    ):
        from ultralytics import YOLO

        path = Path(weights) if weights else WEIGHTS
        if not path.exists():
            import gdown
            path.parent.mkdir(parents=True, exist_ok=True)
            gdown.download(id=GDRIVE_ID, output=str(path), quiet=False)
        self.model = YOLO(str(path))
        self.conf_thr = conf_thr
        self.detect_every = max(1, int(detect_every))
        self.h_alpha = float(h_alpha)
        self.snap_to_paint = bool(snap_to_paint)
        self.last_debug: dict | None = None
        self._frame_i = 0
        self._last_H: np.ndarray | None = None
        self._last_xy: np.ndarray | None = None
        self._last_conf: np.ndarray | None = None

    def _raw(self, bgr):
        r = self.model.predict(bgr, conf=0.25, verbose=False)[0]
        if r.keypoints is None or r.keypoints.xy is None or len(r.keypoints.xy) == 0:
            return None, None
        xy = r.keypoints.xy[0].cpu().numpy().astype(np.float64)
        conf = (
            r.keypoints.conf[0].cpu().numpy().astype(np.float64)
            if r.keypoints.conf is not None
            else np.ones(len(xy), np.float64)
        )
        if self.snap_to_paint:
            xy = snap_keypoints_to_paint(bgr, xy, conf)
        return xy, conf

    def _select_pairs(self, xy, conf):
        keep = []
        for i in range(len(xy)):
            if xy[i, 0] <= 1 or xy[i, 1] <= 1:
                continue
            if i in _CORNER_IDX and conf[i] < 0.95:
                continue
            if i in _CENTRE_IDX and conf[i] >= 0.6:
                keep.append(i)
            elif i in _TRUSTED_IDX and conf[i] >= self.conf_thr:
                keep.append(i)
            elif conf[i] >= 0.9:
                keep.append(i)
        idx = np.array(keep, dtype=int)
        # Need centre + at least one outer landmark when available. Centre-only
        # H fits the circle well but extrapolates badly toward the penalty box
        # (players like ID 148 get shoved onto the touchline).
        if len(idx) < 4:
            return None
        centre = np.array([i for i in idx if i in _CENTRE_IDX], dtype=int)
        outer = np.array([i for i in idx if i not in _CENTRE_IDX], dtype=int)
        if len(centre) >= 4 and len(outer) >= 1:
            idx = np.concatenate([centre, outer])
        elif len(centre) >= 4:
            idx = centre
        src = xy[idx].astype(np.float32)
        dst = PITCH_VERTICES_M[idx].astype(np.float32)
        self.last_debug = {
            "indices": idx.tolist(),
            "names": [_KEYPOINT_NAMES[i] for i in idx],
            "conf": conf[idx].tolist(),
            "image_xy": src,
            "pitch_xy": dst,
        }
        return src, dst, idx

    def _reproj_err(self, H, src, dst) -> np.ndarray:
        pts = np.concatenate([src.astype(np.float64), np.ones((len(src), 1))], 1)
        pred = (np.asarray(H, np.float64) @ pts.T).T
        w = pred[:, 2:3]
        pred = pred[:, :2] / np.where(np.abs(w) < 1e-8, 1e-8, w)
        return np.linalg.norm(pred - dst, axis=1)

    def estimate_H(self, bgr) -> np.ndarray | None:
        self._frame_i += 1
        xy, conf = self._raw(bgr)
        if xy is None:
            return self._last_H
        self._last_xy, self._last_conf = xy, conf
        pairs = self._select_pairs(xy, conf)
        if pairs is None:
            return self._last_H
        src, dst, _ = pairs
        method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
        H, mask = cv2.findHomography(src, dst, method, 2.0)
        if H is None:
            return self._last_H
        if mask is not None and int(mask.sum()) >= 4:
            src = src[mask.ravel().astype(bool)]
            dst = dst[mask.ravel().astype(bool)]
            H2, _ = cv2.findHomography(src, dst, 0)
            if H2 is not None:
                H = H2
        err = self._reproj_err(H, src, dst)
        if float(np.median(err)) > 2.0 or float(np.max(err)) > 8.0:
            return self._last_H
        H = np.asarray(H, np.float64)
        H = H / H[2, 2]
        # Reject H that maps the image centre far off the pitch (bad extrapolation)
        h, w = bgr.shape[:2]
        mid = np.array([w * 0.5, h * 0.55, 1.0], np.float64)
        mid_m = H @ mid
        if abs(mid_m[2]) > 1e-8:
            mid_m = mid_m[:2] / mid_m[2]
            if not (5.0 <= mid_m[0] <= PITCH_L - 5.0 and -5.0 <= mid_m[1] <= PITCH_W + 5.0):
                return self._last_H
        if self._last_H is None:
            self._last_H = H
            return H
        # Elementwise H blend is not a true average; keep it light and favor new H
        a = max(self.h_alpha, 0.65)
        blended = a * H + (1.0 - a) * self._last_H
        blended = blended / blended[2, 2]
        self._last_H = blended
        return blended


class PitchPosSmoother:
    """Light EMA of player pitch positions (optional)."""

    def __init__(self, alpha: float = 0.6, max_jump_m: float = 8.0):
        self.alpha = alpha
        self.max_jump_m = max_jump_m
        self._pos: dict[int, np.ndarray] = {}

    def update(self, tid: int, xy_m: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy_m, np.float64).reshape(2)
        if tid not in self._pos:
            self._pos[tid] = xy
            return xy.astype(np.float32)
        prev = self._pos[tid]
        if np.linalg.norm(xy - prev) > self.max_jump_m:
            xy = prev + 0.4 * (xy - prev)
        sm = self.alpha * xy + (1.0 - self.alpha) * prev
        self._pos[tid] = sm
        return sm.astype(np.float32)

    def prune(self, live_ids: set[int]) -> None:
        for tid in list(self._pos.keys()):
            if tid not in live_ids:
                del self._pos[tid]


def draw_keypoints_debug(bgr, calibrator: PitchKeypointCalibrator) -> np.ndarray:
    vis = bgr.copy()
    xy, conf = calibrator._raw(bgr)
    if xy is None:
        cv2.putText(vis, "pitch keypoints: none", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis
    pairs = calibrator._select_pairs(xy, conf)
    if pairs is None:
        cv2.putText(vis, "pitch keypoints: < 4 visible", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        return vis
    src, dst, idx = pairs
    for (x, y), i in zip(src, idx):
        cv2.circle(vis, (int(x), int(y)), 10, (0, 255, 255), -1)
        cv2.putText(vis, _KEYPOINT_NAMES[int(i)][:12], (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(vis, _KEYPOINT_NAMES[int(i)][:12], (int(x) + 8, int(y) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(vis, f"pitch landmarks used for H: {len(idx)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(vis, f"pitch landmarks used for H: {len(idx)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv2.LINE_AA)
    return vis
