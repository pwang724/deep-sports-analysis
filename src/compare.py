"""Compare the two pose backends on the output of joints.py.

Keeps only on-court players: boxes whose feet land inside the playing area of
the end-on view and whose height is plausible for a player, not a crowd face.
Then reports, per joint and per player size bucket, how far the two models
disagree and how confident each is. Also renders a contact sheet.

    python src/compare.py output/seg320
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

J = ["nose", "l_eye", "r_eye", "l_ear", "r_ear", "l_shoulder", "r_shoulder", "l_elbow", "r_elbow",
     "l_wrist", "r_wrist", "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle"]


def on_court(df: pd.DataFrame, W=1920, H=1080) -> pd.Series:
    """Feet inside the court trapezoid of the end-on broadcast view, plausible height."""
    h = df.y2 - df.y1
    cx = (df.x1 + df.x2) / 2
    fy = df.y2
    # In this broadcast framing the far baseline is at y~0.20H spanning ~0.34W, the near
    # baseline at y~0.89H spanning ~0.84W. Half-width grows linearly between them, plus a
    # margin so a player chasing a wide ball is still counted.
    frac = ((fy - 0.20 * H) / (0.70 * H)).clip(0, 1)
    half_w = (0.17 + 0.26 * frac) * W + 0.05 * W
    inside = (cx - W / 2).abs() < half_w
    return inside & (fy > 0.19 * H) & (fy < 1.0 * H) & (h > 60) & (h < 600)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    a = p.parse_args()
    d = Path(a.run_dir)
    df = pd.read_parquet(d / "joints.parquet")
    df["h"] = df.y2 - df.y1
    df = df[on_court(df)]
    r = df[df.backend == "rfdetr"].set_index(["frame", "track_id"])
    v = df[df.backend == "vitpose"].set_index(["frame", "track_id"])
    idx = r.index.intersection(v.index)
    r, v = r.loc[idx], v.loc[idx]
    print(f"on-court paired rows: {len(idx)}   frames: {r.index.get_level_values(0).nunique()}")

    report = {}
    for lo, hi, name in [(60, 160, "far player"), (160, 600, "near player")]:
        m = (r.h >= lo) & (r.h < hi)
        if m.sum() == 0:
            continue
        print(f"\n{name}  n={int(m.sum())}  median box height {r.loc[m, 'h'].median():.0f}px")
        print(f"{'joint':12s} {'med px':>7s} {'% of h':>7s} {'rf conf':>8s} {'vp conf':>8s}")
        rows = {}
        for j in J:
            dpx = np.hypot(r.loc[m, f"{j}_x"] - v.loc[m, f"{j}_x"], r.loc[m, f"{j}_y"] - v.loc[m, f"{j}_y"])
            dn = 100 * (dpx / r.loc[m, "h"]).median()
            rows[j] = dict(med_px=float(dpx.median()), pct_h=float(dn), rf_conf=float(r.loc[m, f"{j}_c"].mean()), vp_conf=float(v.loc[m, f"{j}_c"].mean()))
            print(f"{j:12s} {rows[j]['med_px']:7.1f} {rows[j]['pct_h']:7.1f} {rows[j]['rf_conf']:8.2f} {rows[j]['vp_conf']:8.2f}")
        report[name] = rows

    # track continuity for on-court players
    tl = df[df.backend == "rfdetr"].groupby("track_id").frame.agg(["count", "min", "max"]).sort_values("count", ascending=False)
    print("\nlongest on-court tracks (frames):")
    print(tl.head(6).to_string())
    report["tracks"] = tl.head(10).reset_index().to_dict("records")
    (d / "compare.json").write_text(json.dumps(report, indent=2))

    # contact sheet: 6 evenly spaced annotated frames
    cap = cv2.VideoCapture(str(d / "annotated.mp4"))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    tiles = []
    for k in np.linspace(0, n - 1, 6).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k))
        ok, f = cap.read()
        if ok:
            tiles.append(cv2.resize(f, (960, 540)))
    if tiles:
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(str(d / "sheet.png"), sheet)
        print(f"\nwrote {d / 'sheet.png'}")


if __name__ == "__main__":
    main()
