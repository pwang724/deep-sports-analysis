"""One converter per public dataset: data/sources/<name> -> data/labels/<name> (dsa.data.schema).

Each module has `convert() -> {table: DataFrame}`; `run(name)` validates and
writes it. `python -m dsa.data.convert.<name>` converts one source,
`python -m dsa.data.convert` all of them and prints the row counts.

Conventions shared by every converter: the labeler is the source name; boxes
are stored as x1, y1, x2, y2; media paths are relative to data/ and recorded
whether or not the file has been downloaded; tables a source does not label
are not written.
"""
from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import pandas as pd

from dsa.data import schema
from dsa.data.paths import DATA

NAMES = ["tracknet", "tennis_court_detector", "racketvision", "coco_wholebody", "tennis_segmentation",
         "tennis_player_actions", "e2e_spot", "f3set"]


def rel(path: Path) -> str:
    """A path under data/ as stored in `media`."""
    return str(path.relative_to(DATA))


def xyxy(xywh) -> list[float]:
    x, y, w, h = map(float, xywh)
    return [x, y, x + w, y + h]


def frame_row(source: str, media_id: str, frame: int, split: str, media: str, fps: float, w: int, h: int) -> dict:
    return {"sample": schema.sample_id(source, media_id, frame), "source": source, "split": split, "media": media,
            "frame": frame, "fps": fps, "width": w, "height": h}


def span(source: str, split: str, media: str, start: int, end: int, fps: float, covers: str) -> dict:
    return {"source": source, "split": split, "media": media, "start": start, "end": end, "fps": fps,
            "covers": covers, "labeler": source}


def events_frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=schema.TABLES["events"])
    return df.astype({c: "string" for c in ("type", "side", "hand", "technique", "direction", "outcome")})


def counts(tables: dict[str, pd.DataFrame]) -> dict[str, int]:
    return {t: len(tables[t]) if t in tables else 0
            for t in ("frames", "people", "rackets", "ball", "court", "events", "spans")}


def run(name: str) -> dict[str, int]:
    tables = importlib.import_module(f"dsa.data.convert.{name}").convert()
    out = schema.write(name, tables)
    for stale in set(schema.TABLES) - set(tables):
        (out / f"{stale}.parquet").unlink(missing_ok=True)
    n = counts(tables)
    split = tables["frames"] if "frames" in tables else tables["events"]
    print(f"{name}: " + ", ".join(f"{t} {v}" for t, v in n.items() if v)
          + "  splits " + ", ".join(f"{s} {k}" for s, k in split["split"].value_counts().sort_index().items()))
    return n


def empty_kp_list(n: int = len(schema.KEYPOINTS)) -> list[float]:
    return schema.flat(schema.empty_kp(n))


def vis_map(v, table: dict) -> float:
    """A source's visibility flag to the schema's (2 / 1 / 0 / NaN), via `table`."""
    return table.get(int(v), np.nan)
