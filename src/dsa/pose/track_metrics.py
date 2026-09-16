"""Compare trackers on an unlabelled end-on broadcast clip.

There are no track labels, so the two players are located per frame with a
geometric rule that is only used for scoring: the near player is the box with
the lowest feet in the centre band of the frame, the far player the box with
the highest feet in the centre band of the far court. Ball kids and line
judges stand outside those bands in this framing.

Metrics per tracker: how often each player is found, how many distinct IDs
each player is given (1 is perfect), boxes per frame, and seconds per frame.
"""
from __future__ import annotations

import pandas as pd

# Fractions of frame width / height, measured on the US Open end-on framing.
# Near player: feet in the lower 40% of the frame, x in the middle half.
# Far player: feet just below the far baseline (17-32% of height), x in the
# middle 28%; ball kids behind the baseline sit at ~15% height and further out.
NEAR_BAND = dict(x=(0.25, 0.75), y2=(0.60, 1.0))
FAR_BAND = dict(x=(0.36, 0.64), y2=(0.17, 0.32))


def _in_band(df: pd.DataFrame, band: dict, W: int, H: int) -> pd.Series:
    cx = (df.x1 + df.x2) / 2 / W
    fy = df.y2 / H
    return (cx >= band["x"][0]) & (cx <= band["x"][1]) & (fy >= band["y2"][0]) & (fy <= band["y2"][1])


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def assign_players(rows: pd.DataFrame, W: int, H: int, min_iou: float = 0.3) -> pd.DataFrame:
    """One row per (frame, player) with the track id chosen for that player, or none.

    Within the player's band, the box that overlaps the previous frame's choice
    is preferred, so the assignment follows the person rather than flipping
    between people who share the band; with no overlap it falls back to the
    lowest feet (near) or highest feet (far).
    """
    last = {"near": None, "far": None}
    out = []
    for frame, g in rows.groupby("frame"):
        for player, band, pick in (("near", NEAR_BAND, "max"), ("far", FAR_BAND, "min")):
            cand = g[_in_band(g, band, W, H)]
            chosen = None
            if len(cand):
                boxes = cand[["x1", "y1", "x2", "y2"]].to_numpy()
                if last[player] is not None:
                    ious = [_iou(last[player], b) for b in boxes]
                    if max(ious) >= min_iou:
                        chosen = cand.iloc[int(pd.Series(ious).idxmax())]
                if chosen is None:
                    chosen = cand.loc[cand.y2.idxmax() if pick == "max" else cand.y2.idxmin()]
                last[player] = chosen[["x1", "y1", "x2", "y2"]].to_numpy(float)
            out.append({"frame": frame, "player": player, "track_id": int(chosen.track_id) if chosen is not None else None})
    return pd.DataFrame(out)


def id_changes(track_ids: pd.Series) -> int:
    """Number of times the id changes between consecutive frames where the player was found."""
    found = track_ids.dropna().astype(int).tolist()
    return sum(1 for a, b in zip(found, found[1:]) if a != b)


def tracker_report(rows: pd.DataFrame, frames: list[int], W: int, H: int) -> dict:
    """Coverage and identity stability for both players over `frames`."""
    assigned = assign_players(rows, W, H) if len(rows) else pd.DataFrame(columns=["frame", "player", "track_id"])
    report = {"frames": len(frames), "boxes_per_frame": len(rows) / max(len(frames), 1), "tracks": int(rows.track_id.nunique()) if len(rows) else 0}
    for player in ("near", "far"):
        a = assigned[assigned.player == player].set_index("frame").reindex(frames)
        report[f"{player}_found"] = float(a.track_id.notna().mean()) if len(a) else 0.0
        report[f"{player}_ids"] = int(a.track_id.dropna().nunique())
        report[f"{player}_id_changes"] = id_changes(a.track_id)
    return report
