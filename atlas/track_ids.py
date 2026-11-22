"""Keep stable display track IDs across brief detection gaps / class flips."""
from __future__ import annotations

import numpy as np


class TrackIdStabilizer:
    """If a new BoT-SORT id appears near a recently lost track, reuse the old id.

    Helps when the GK (or any player) flickers at the frame edge and the tracker
    opens a fresh track id.
    """

    def __init__(self, max_dist_m: float = 7.0, max_gap_frames: int = 75):
        self.max_dist_m = float(max_dist_m)
        self.max_gap_frames = int(max_gap_frames)
        self._frame = 0
        self._last_pos: dict[int, tuple[np.ndarray, int]] = {}
        self._alias: dict[int, int] = {}

    def remap(self, objects: list[dict]) -> None:
        """Mutate ``objects[*]['tid']`` in place to stable ids."""
        self._frame += 1
        # Resolve existing aliases
        for o in objects:
            rid = int(o["tid"])
            o["tid"] = int(self._alias.get(rid, rid))

        live = {int(o["tid"]) for o in objects}
        # New raw ids that are not yet in history → try reclaim a lost neighbour
        for o in objects:
            tid = int(o["tid"])
            if o.get("x_m") is None or o.get("y_m") is None:
                continue
            xy = np.array([float(o["x_m"]), float(o["y_m"])], np.float64)
            if tid in self._last_pos:
                continue  # already known canonical id
            # brand-new id this session (or never positioned)
            best_id, best_d = None, self.max_dist_m
            for old_id, (pos, f) in self._last_pos.items():
                if old_id in live:
                    continue  # still active under another detection
                gap = self._frame - f
                if gap < 1 or gap > self.max_gap_frames:
                    continue
                d = float(np.linalg.norm(xy - pos))
                if d < best_d:
                    best_d, best_id = d, old_id
            if best_id is not None:
                self._alias[tid] = best_id
                o["tid"] = best_id
                live.add(best_id)

        # Refresh positions for live tracks
        for o in objects:
            if o.get("x_m") is None or o.get("y_m") is None:
                continue
            tid = int(o["tid"])
            self._last_pos[tid] = (
                np.array([float(o["x_m"]), float(o["y_m"])], np.float64),
                self._frame,
            )

        # Drop very stale entries
        stale = [
            t
            for t, (_, f) in self._last_pos.items()
            if self._frame - f > self.max_gap_frames * 2
        ]
        for t in stale:
            del self._last_pos[t]
