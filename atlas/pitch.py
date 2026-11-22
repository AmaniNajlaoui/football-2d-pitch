"""Pitch lines → homography H (image → 105×68 m pitch).

Preferred: SoccerNet DeepLab pitch segmentation (official calibration baseline).
Fallback: Roboflow pitch keypoints → green field corners.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "weights"
PITCH_PTH = WEIGHTS / "soccer_pitch_segmentation.pth"
GDRIVE_ID = "1dbN7LdMV03BR1Eda8n7iKNIyYp9r07sM"

PITCH_L, PITCH_W = 105.0, 68.0

# Ground lines we trust for H (skip goals / unknown / arcs)
_TRUSTED_LINES = {
    "Side line top",
    "Side line bottom",
    "Side line left",
    "Side line right",
    "Middle line",
    "Big rect. left bottom",
    "Big rect. left main",
    "Big rect. left top",
    "Big rect. right bottom",
    "Big rect. right main",
    "Big rect. right top",
    "Small rect. left bottom",
    "Small rect. left main",
    "Small rect. left top",
    "Small rect. right bottom",
    "Small rect. right main",
    "Small rect. right top",
}


def download_pitch_weights():
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    if not (WEIGHTS / "mean.npy").exists() or not (WEIGHTS / "std.npy").exists():
        import urllib.request
        base = "https://raw.githubusercontent.com/SoccerNet/sn-gamestate/main/pretrained_models/calibration"
        for name in ("mean.npy", "std.npy"):
            urllib.request.urlretrieve(f"{base}/{name}", WEIGHTS / name)
    if PITCH_PTH.exists() and PITCH_PTH.stat().st_size > 1_000_000:
        return PITCH_PTH
    import gdown
    gdown.download(id=GDRIVE_ID, output=str(PITCH_PTH), quiet=False)
    return PITCH_PTH


class SoccerNetPitchSeg:
    """SoccerNet calibration DeepLab (29 line classes).

    Matches SoccerNet/sn-calibration SegmentationNetwork: BN eps=1e-3 is required
    for the public weights to work (default eps=1e-5 predicts all-background).
    """

    def __init__(self):
        download_pitch_weights()
        self.mean = np.load(WEIGHTS / "mean.npy")
        self.std = np.load(WEIGHTS / "std.npy")
        self.width, self.height = 640, 360
        # weights_backbone=None → random init, then overwritten by SoccerNet ckpt
        model = deeplabv3_resnet50(
            weights=None, weights_backbone=None, num_classes=29, aux_loss=False
        )
        model = nn.DataParallel(model)
        # SoccerNet training used BN eps=1e-3 / momentum=0.1
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.1
        ckpt = torch.load(PITCH_PTH, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.last_mask = None
        self.last_ext = None

    @torch.no_grad()
    def lines_mask(self, bgr):
        # Official baseline expects BGR from cv2.imread
        img = cv2.resize(bgr, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        x = np.asarray(img, np.float32) / 255.0
        x = (x - self.mean) / self.std
        x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
        out = self.model(x)["out"][0].cpu().numpy()
        mask = np.argmax(out, axis=0).astype(np.uint8)
        self.last_mask = mask
        return mask


def _sn_to_map(p3):
    """SoccerNet centred pitch (m) → our map coords [0,105]×[0,68]."""
    return np.array([p3[0] + PITCH_L / 2.0, p3[1] + PITCH_W / 2.0], np.float32)


def _order_pair(img_a, img_b, world_a, world_b):
    """Order 2 image extremities to match 2 world points (sort by dominant axis)."""
    img = np.stack([img_a, img_b]).astype(np.float64)
    world = np.stack([world_a, world_b]).astype(np.float64)
    if abs(img[0, 0] - img[1, 0]) >= abs(img[0, 1] - img[1, 1]):
        i_order = np.argsort(img[:, 0])
        w_order = np.argsort(world[:, 0])
    else:
        i_order = np.argsort(img[:, 1])
        w_order = np.argsort(world[:, 1])
    return img[i_order], world[w_order]


def homography_from_soccernet_mask(mask, img_wh, seg: SoccerNetPitchSeg | None = None):
    """Line extremities from SoccerNet DeepLab → H (image → metres).

    Side-line extremities are often only a *visible segment*, not the true pitch
    corners — so we prefer Middle line + box lines, and sanity-check H.
    """
    from vendor.sn_calibration_baseline.detect_extremities import (
        generate_class_synthesis,
        get_line_extremities,
    )
    from vendor.sn_calibration_baseline.soccerpitch import SoccerPitch

    w, h = img_wh
    skeletons = generate_class_synthesis(mask, radius=6)
    ext = get_line_extremities(
        skeletons, maxdist=40, width=mask.shape[1], height=mask.shape[0]
    )
    if seg is not None:
        seg.last_ext = ext
    field = SoccerPitch(pitch_length=PITCH_L, pitch_width=PITCH_W)

    # Prefer local/stable lines. Full sidelines → corners is unreliable when cropped.
    preferred = [
        "Middle line",
        "Big rect. left main",
        "Big rect. left top",
        "Big rect. left bottom",
        "Big rect. right main",
        "Big rect. right top",
        "Big rect. right bottom",
        "Small rect. left main",
        "Small rect. left top",
        "Small rect. left bottom",
        "Small rect. right main",
        "Small rect. right top",
        "Small rect. right bottom",
        "Side line top",
        "Side line bottom",
    ]

    src, dst = [], []
    for name in preferred:
        if name not in ext or name not in field.line_extremities:
            continue
        pts = ext[name]
        if len(pts) < 2:
            continue
        wa, wb = field.line_extremities[name]
        if abs(float(wa[2])) > 1e-3 or abs(float(wb[2])) > 1e-3:
            continue
        ia = np.array([pts[0]["x"] * w, pts[0]["y"] * h], np.float64)
        ib = np.array([pts[1]["x"] * w, pts[1]["y"] * h], np.float64)
        # skip tiny segments (noise)
        if np.linalg.norm(ia - ib) < 25:
            continue
        img_o, world_o = _order_pair(ia, ib, _sn_to_map(wa), _sn_to_map(wb))
        src.extend(img_o.tolist())
        dst.extend(world_o.tolist())

    if len(src) < 4:
        return None
    src = np.float32(src)
    dst = np.float32(dst)
    method = getattr(cv2, "USAC_MAGSAC", cv2.RANSAC)
    H, mask_h = cv2.findHomography(src, dst, method, 5.0)
    if H is None:
        return None
    if mask_h is not None and int(mask_h.sum()) < 4:
        return None

    # Sanity: Middle line midpoint should land near x≈52.5 if that line was used
    if "Middle line" in ext and len(ext["Middle line"]) >= 2:
        pts = ext["Middle line"]
        mid = np.float32([
            [
                0.5 * (pts[0]["x"] + pts[1]["x"]) * w,
                0.5 * (pts[0]["y"] + pts[1]["y"]) * h,
            ]
        ])
        from atlas.minimap import apply_H

        xy = apply_H(H, mid)[0]
        if abs(float(xy[0]) - PITCH_L / 2.0) > 12.0:
            return None
        if not (-5.0 <= float(xy[1]) <= PITCH_W + 5.0):
            return None
    return H


def homography_from_green(bgr):
    """Fallback H: green field contour → rectangle pitch."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (35, 40, 40), (90, 255, 255))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.05 * bgr.shape[0] * bgr.shape[1]:
        return None
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) != 4:
        rect = cv2.minAreaRect(c)
        box = cv2.boxPoints(rect)
    else:
        box = approx.reshape(4, 2).astype(np.float32)
    box = _order_corners(box)
    dst = np.array([[0, 0], [PITCH_L, 0], [PITCH_L, PITCH_W], [0, PITCH_W]], np.float32)
    H, _ = cv2.findHomography(box, dst, cv2.RANSAC, 5.0)
    return H


def _order_corners(pts):
    pts = np.array(pts, np.float32)
    s = pts.sum(1)
    d = np.diff(pts, axis=1).ravel()
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl]).astype(np.float32)


_PITCH_KP = None  # lazy PitchKeypointCalibrator


def get_pitch_keypoint_calibrator(reset: bool = False):
    global _PITCH_KP
    if reset or _PITCH_KP is None:
        from atlas.pitch_keypoints import PitchKeypointCalibrator
        _PITCH_KP = PitchKeypointCalibrator(detect_every=1, h_alpha=0.75, conf_thr=0.7)
    return _PITCH_KP


def estimate_H(
    bgr,
    seg: SoccerNetPitchSeg | None = None,
    use_keypoints: bool = True,
    prefer_soccernet: bool = False,
):
    """Roboflow keypoints (preferred) → SoccerNet lines → green fallback.

    SoccerNet DeepLab extremities often map a *visible segment* to full-pitch
    corners, which stretches the minimap. Keypoints are more stable on broadcast.
    """
    H = None
    if prefer_soccernet and seg is not None:
        try:
            mask = seg.lines_mask(bgr)
            H = homography_from_soccernet_mask(mask, (bgr.shape[1], bgr.shape[0]), seg=seg)
        except Exception:
            H = None
    if H is None and use_keypoints:
        try:
            H = get_pitch_keypoint_calibrator().estimate_H(bgr)
        except Exception:
            H = None
    if H is None and seg is not None:
        try:
            mask = seg.lines_mask(bgr)
            H = homography_from_soccernet_mask(mask, (bgr.shape[1], bgr.shape[0]), seg=seg)
        except Exception:
            H = None
    if H is None:
        H = homography_from_green(bgr)
    return H


def colorize_soccernet_mask(mask: np.ndarray) -> np.ndarray:
    """RGB visualisation of SoccerNet line classes."""
    from vendor.sn_calibration_baseline.soccerpitch import SoccerPitch

    h, w = mask.shape
    vis = np.zeros((h, w, 3), np.uint8)
    for k, name in enumerate(SoccerPitch.lines_classes):
        color = SoccerPitch.palette.get(name, (0, 0, 0))
        # palette is RGB; OpenCV wants BGR
        vis[mask == (k + 1)] = (color[2], color[1], color[0])
    return vis
