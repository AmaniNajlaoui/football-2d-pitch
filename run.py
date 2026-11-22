#!/usr/bin/env python3
"""SoccerNet-style demo: detect → track → team colour → pitch H → 2D map.

Not Atlas. Same idea as SoccerNet Game State Reconstruction.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from atlas.ball import BallTracker, draw_ball_trail
from atlas.minimap import apply_H, draw_empty_pitch, stamp_objects
from atlas.pitch import SoccerNetPitchSeg, estimate_H, get_pitch_keypoint_calibrator
from atlas.pitch_keypoints import PitchPosSmoother
from atlas.teams import assign_teams, jersey_lab, reset_team_model
from atlas.track_ids import TrackIdStabilizer

# Team 0 = white kit, team 1 = red kit (BGR). Ball = yellow so it ≠ red team.
TEAM_COLORS = {0: (255, 255, 255), 1: (0, 0, 255), None: (0, 215, 255)}
GK_COLOR = (0, 200, 0)  # green box when YOLO class is goalkeeper
BALL_COLOR = (0, 220, 255)
BALL_NAMES = {"ball", "sports ball"}
PERSON_NAMES = {"person", "player", "referee", "goalkeeper", "goalie"}
GK_NAMES = {"goalkeeper", "goalie"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Path to a football clip")
    p.add_argument("--out", default=str(ROOT / "outputs" / "demo.mp4"))
    p.add_argument(
        "--weights",
        default=str(ROOT / "yolov8" / "best.pt"),
        help="Ultralytics YOLO weights (default: Colab YOLOv8x best.pt)",
    )
    p.add_argument(
        "--tracker",
        default=str(ROOT / "cfg" / "botsort_football.yaml"),
        help="Ultralytics tracker YAML (default: longer-buffer football BoT-SORT)",
    )
    p.add_argument("--max-frames", type=int, default=300)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--no-soccernet-pitch", action="store_true")
    p.add_argument(
        "--H",
        default=None,
        help="Fixed homography .npy from scripts/pick_homography.py (skips auto H)",
    )
    return p.parse_args()


def load_yolo(weights: str):
    from ultralytics import YOLO
    return YOLO(weights)


def main():
    args = parse_args()
    video = Path(args.video)
    if not video.exists():
        sys.exit(f"Video not found: {video}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = load_yolo(args.weights)
    names = model.names
    reset_team_model()

    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or args.max_frames
    n = min(n, args.max_frames)

    seg = None
    if not args.no_soccernet_pitch:
        try:
            print("Loading SoccerNet pitch segmentation…")
            seg = SoccerNetPitchSeg()
        except Exception as e:
            print(f"SoccerNet pitch weights not used ({e}). Green-field fallback.")

    pitch_blank, xy = draw_empty_pitch(h=H, w=int(H * 105 / 68))
    map_w = pitch_blank.shape[1]
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W + map_w, H),
    )

    lab_hist = defaultdict(list)
    last_H = None
    fixed_H = None
    # Light smooth only — heavy EMA lagged behind the camera and looked "wrong"
    pos_smooth = PitchPosSmoother(alpha=0.75, max_jump_m=10.0)
    id_stable = TrackIdStabilizer(max_dist_m=7.0, max_gap_frames=75)
    ball_tr = BallTracker(conf_thr=0.05, trail_len=500, max_coast_frames=12, max_jump_m=4.5)
    if args.H:
        fixed_H = np.load(args.H)
        last_H = fixed_H
        print(f"Using fixed keypoint H from {args.H}")
    else:
        get_pitch_keypoint_calibrator(reset=True)
        print("Auto H: Roboflow pitch keypoints (SoccerNet lines = fallback only)")
    print(f"Running {n} frames from {video.name} …")

    for i in tqdm(range(n)):
        ok, frame = cap.read()
        if not ok:
            break
        # People track at normal conf. Ball is a separate low-conf predict:
        # BoT-SORT replaces boxes with tracked objs and drops tiny low-conf balls.
        results = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False,
        )
        r = results[0]
        vis = frame.copy()
        objects = []
        frame_labs = {}
        ball_det_img = ball_tr.detect_image(model, frame, imgsz=args.imgsz)

        if r.boxes is not None and r.boxes.id is not None:
            xyxy = r.boxes.xyxy.cpu().numpy()
            cls = r.boxes.cls.cpu().numpy().astype(int)
            tids = r.boxes.id.cpu().numpy().astype(int)
            for box, c, tid in zip(xyxy, cls, tids):
                x1, y1, x2, y2 = box
                cname = str(names.get(int(c), "")).lower()
                is_ball = cname in BALL_NAMES
                is_ref = cname in {"referee"}
                is_gk = cname in GK_NAMES
                if is_ball:
                    continue
                if cname not in PERSON_NAMES:
                    continue
                if is_ref:
                    kind = "referee"
                elif is_gk:
                    kind = "goalkeeper"
                else:
                    kind = "player"
                # Ground point: slightly above bbox bottom (reduces shoe/shadow bias)
                foot_y = y2 - 0.02 * max(y2 - y1, 1.0)
                foot = np.array([[(x1 + x2) / 2.0, foot_y]], np.float32)
                # Outfield only for team colour (GK / ref kits pollute white/red)
                if kind == "player":
                    feat = jersey_lab(frame, box)
                    if feat is not None:
                        lab_hist[int(tid)].append(feat)
                        frame_labs[int(tid)] = feat
                objects.append(
                    dict(kind=kind, tid=int(tid), box=box, foot=foot[0], team=None)
                )

        means = {t: np.mean(v, axis=0) for t, v in lab_hist.items() if len(v) >= 3}
        teams = assign_teams(means) if means else {}
        for o in objects:
            if o["kind"] == "player":
                o["team"] = teams.get(o["tid"])
                if o["team"] is None and o["tid"] in teams and teams[o["tid"]] is None:
                    o["kind"] = "referee"

        # Moving camera → keypoints first (SoccerNet extremities stretch the map)
        if fixed_H is None:
            Hmat = estimate_H(frame, seg, use_keypoints=True, prefer_soccernet=False)
            if Hmat is not None:
                last_H = Hmat

        # Pitch positions first (needed for ID reclaim by proximity)
        for o in objects:
            if last_H is not None:
                p = apply_H(last_H, o["foot"].reshape(1, 2))[0]
                p = np.array([
                    float(np.clip(p[0], -3.0, 108.0)),
                    float(np.clip(p[1], -3.0, 71.0)),
                ], np.float32)
                o["x_m"], o["y_m"] = float(p[0]), float(p[1])
            else:
                o["x_m"] = o["y_m"] = None

        # Reuse nearby lost track ids (stops GK ID flicker on brief misses)
        id_stable.remap(objects)

        live_ids = set()
        for o in objects:
            if o["x_m"] is not None and o["kind"] != "ball":
                sm = pos_smooth.update(int(o["tid"]), np.array([o["x_m"], o["y_m"]], np.float32))
                o["x_m"], o["y_m"] = float(sm[0]), float(sm[1])
                live_ids.add(int(o["tid"]))
            x1, y1, x2, y2 = o["box"]
            if o["kind"] == "ball":
                color = BALL_COLOR
            elif o["kind"] == "goalkeeper":
                color = GK_COLOR
            else:
                color = TEAM_COLORS.get(o["team"], (200, 200, 200))
            # Dark outline so white boxes stay visible on the pitch
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (20, 20, 20), 3)
            cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            if o["kind"] == "ball":
                label = "ball"
            elif o["kind"] == "referee":
                label = f"ref {o['tid']}"
            elif o["kind"] == "goalkeeper":
                label = f"GK {o['tid']}"
            else:
                label = f"ID {o['tid']}"
            cv2.putText(vis, label, (int(x1), max(15, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3, cv2.LINE_AA)
            cv2.putText(vis, label, (int(x1), max(15, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        pos_smooth.prune(live_ids)

        # --- Ball path on 2D map (low-conf det + coast through misses) ---
        ball_xy_m = None
        if ball_det_img is not None and last_H is not None:
            img_xy, bconf = ball_det_img
            proj = apply_H(last_H, img_xy.reshape(1, 2).astype(np.float32))[0]
            # Off the FIFA pitch → stands, board, or a bad homography. Do not lock onto it.
            if -2.0 <= float(proj[0]) <= 107.0 and -2.0 <= float(proj[1]) <= 70.0:
                ball_xy_m = proj
                ball_tr.last_img_xy = img_xy
                ix, iy = int(img_xy[0]), int(img_xy[1])
                cv2.circle(vis, (ix, iy), 10, (20, 20, 20), 3)
                cv2.circle(vis, (ix, iy), 10, BALL_COLOR, 2)
                cv2.putText(vis, f"ball {bconf:.2f}", (ix + 12, iy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 3, cv2.LINE_AA)
                cv2.putText(vis, f"ball {bconf:.2f}", (ix + 12, iy - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, BALL_COLOR, 1, cv2.LINE_AA)
        ball_pos = ball_tr.update(None if ball_xy_m is None else np.asarray(ball_xy_m, np.float64))

        mini = stamp_objects(pitch_blank, xy, objects)
        draw_ball_trail(mini, xy, list(ball_tr.trail), color=BALL_COLOR)
        if ball_pos is not None:
            p = xy(float(np.clip(ball_pos[0], 0, 105)), float(np.clip(ball_pos[1], 0, 68)))
            cv2.circle(mini, p, 12, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(mini, p, 8, BALL_COLOR, -1)
            cv2.circle(mini, p, 8, (20, 20, 20), 1, cv2.LINE_AA)
        if mini.shape[0] != H:
            mini = cv2.resize(mini, (map_w, H))
        if fixed_H is not None:
            h_tag = "manual H"
        else:
            h_tag = "keypoint H"
        banner = f"YOLOv8x + BoT-SORT + ball trail + {h_tag}"
        cv2.putText(vis, banner, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        writer.write(np.hstack([vis, mini]))

    cap.release()
    writer.release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
