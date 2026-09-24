"""The one label format every source is converted to, and the trainer reads.

One directory per source, data/labels/<source>/, holding up to six Parquet
tables. Every table has a `sample` key, "<source>/<media id>/<frame>" (frame
-1 for a still image), and a `labeler` column naming who made the label: a
dataset name for human labels ("tracknet"), or a model or Astra with a version
("wasb", "astra:gpt-6-astra:medium"). A missing row or a NaN value means "not
labelled" and is masked out of the loss; it never means "absent". Absence is
stated explicitly (a ball with visible = False, a keypoint with vis = 0).

  frames    sample, source, split, media, frame, fps, width, height
            One row per labelled frame. `media` is the image or video path
            relative to data/; `frame` indexes the video (-1 for an image).
  people    sample, person, track, box (x1, y1, x2, y2), kp (30 x 3), stroke, labeler
            One row per person. kp rows are (x, y, vis) in KEYPOINTS order;
            vis 2 = visible, 1 = labelled but occluded, 0 = not in view,
            NaN = not labelled. `stroke` is the coarse class of a single image
            where the source gives one (forehand, backhand, serve, ready).
  rackets   sample, racket, box, kp (5 x 3), person, labeler
            Rackets as labelled, before they are attached to a person
            (`person` NaN until then).
  ball      sample, x, y, visible, labeler
  court     sample, kp (14 x 3), labeler             COURT_POINTS order
  scene     sample, view, in_play, labeler           booleans, NaN = not labelled
  events    source, media, frame, fps, type, side, hand, technique, direction, outcome, labeler
            Shots and bounces in video time. type: serve | shot | bounce;
            side: near | far; hand: forehand | backhand; technique: gs | slice
            | volley | smash | drop | lob; direction: CC | DL | DM | II | IO
            (shots) or T | B | W (serves). Unlabelled fields are null.

Coordinates are pixels of the frame at `width` x `height`. Arrays are stored
as flat float lists and read back with `keypoints()`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dsa.data.paths import LABELS

BODY = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle"]
FEET = ["left_big_toe", "left_small_toe", "left_heel", "right_big_toe", "right_small_toe", "right_heel"]
RACKET = ["racket_tip", "racket_head_bottom", "racket_handle", "racket_left", "racket_right"]
KEYPOINTS = BODY + ["neck", "head_top"] + FEET + RACKET          # 30 per person
KP = {name: i for i, name in enumerate(KEYPOINTS)}

# TennisCourtDetector order: line intersections on the reference court.
COURT_POINTS = [
    "far_baseline_left", "far_baseline_right", "near_baseline_left", "near_baseline_right",
    "far_singles_left", "near_singles_left", "far_singles_right", "near_singles_right",
    "far_service_left", "far_service_right", "near_service_left", "near_service_right",
    "far_t", "near_t",
]

TABLES = {
    "frames": ["sample", "source", "split", "media", "frame", "fps", "width", "height"],
    "people": ["sample", "person", "track", "box", "kp", "stroke", "labeler"],
    "rackets": ["sample", "racket", "box", "kp", "person", "labeler"],
    "ball": ["sample", "x", "y", "visible", "labeler"],
    "court": ["sample", "kp", "labeler"],
    "scene": ["sample", "view", "in_play", "labeler"],
    "events": ["source", "media", "frame", "fps", "type", "side", "hand", "technique", "direction", "outcome",
               "labeler"],
}
ARRAYS = {("people", "kp"): (len(KEYPOINTS), 3), ("rackets", "kp"): (5, 3), ("court", "kp"): (14, 3),
          ("people", "box"): (4,), ("rackets", "box"): (4,)}
SPLITS = {"train", "val", "test"}


def sample_id(source: str, media_id: str, frame: int = -1) -> str:
    return f"{source}/{media_id}/{frame}"


def empty_kp(n: int = len(KEYPOINTS)) -> np.ndarray:
    return np.full((n, 3), np.nan)


def flat(a) -> list[float] | None:
    return None if a is None else np.asarray(a, float).ravel().tolist()


def keypoints(values, n: int = len(KEYPOINTS)) -> np.ndarray:
    """A stored flat list back to an (n, 3) array."""
    return np.asarray(values, float).reshape(n, 3)


def validate(table: str, df: pd.DataFrame) -> None:
    missing = set(TABLES[table]) - set(df.columns)
    if missing:
        raise ValueError(f"{table}: missing columns {sorted(missing)}")
    if table != "events" and df["sample"].isna().any():
        raise ValueError(f"{table}: empty sample key")
    if table == "frames":
        if df["sample"].duplicated().any():
            raise ValueError("frames: duplicate samples")
        if not set(df["split"]) <= SPLITS:
            raise ValueError(f"frames: splits {set(df['split']) - SPLITS}")
    for (t, col), shape in ARRAYS.items():
        if t == table:
            n = int(np.prod(shape))
            bad = df[col].map(lambda v: v is not None and len(v) != n)
            if bad.any():
                raise ValueError(f"{table}.{col}: {int(bad.sum())} rows not of length {n}")
    if "labeler" in df and df["labeler"].isna().any():
        raise ValueError(f"{table}: rows without a labeler")


def write(source: str, tables: dict[str, pd.DataFrame], root: Path = LABELS) -> Path:
    """Validate and write a source's tables to data/labels/<source>/; returns the directory."""
    out = root / source
    out.mkdir(parents=True, exist_ok=True)
    frames = tables.get("frames")
    for name, df in tables.items():
        if name not in TABLES:
            raise ValueError(f"unknown table {name}")
        validate(name, df)
        if frames is not None and name not in ("frames", "events"):
            orphans = set(df["sample"]) - set(frames["sample"])
            if orphans:
                raise ValueError(f"{name}: {len(orphans)} samples not in frames, e.g. {next(iter(orphans))}")
        df[TABLES[name] + [c for c in df.columns if c not in TABLES[name]]].to_parquet(out / f"{name}.parquet",
                                                                                        index=False)
    return out


def read(source: str, table: str, root: Path = LABELS) -> pd.DataFrame:
    return pd.read_parquet(root / source / f"{table}.parquet")
