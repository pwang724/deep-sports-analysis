"""TennisSegmentation -> frames, people, ball.

197 broadcast frames with segmentation masks, read with
dsa.data.tennis_segmentation (see its docstring for the dropped degenerate
player masks and the ball centre). The images live only inside the three
Parquet shards under data/, so media is the shard and frame the row within it
(sample "tennis_segmentation/train-00000-of-00003/<row>"). The sample names in
TennisSegmentation.json cover only 49 of the 197 frames and are not used.

People: the far (playerTop) and near (playerBottom) players as boxes, track
"far" / "near", no keypoints. Ball: the mask centre, always visible (every
frame has a ball mask). Split: test.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from dsa.data.convert import empty_kp_list, frame_row, rel, run
from dsa.data.paths import SOURCES
from dsa.data.tennis_segmentation import load

NAME = "tennis_segmentation"
ROOT = SOURCES / "tennis_segmentation"
TRACK = {"top": "far", "bottom": "near"}


def convert() -> dict[str, pd.DataFrame]:
    shards = [(p, pq.ParquetFile(p).metadata.num_rows) for p in sorted((ROOT / "data").glob("*.parquet"))
              if not p.name.startswith("._")]
    where = [(p, row) for p, n in shards for row in range(n)]
    frames, people, ball = [], [], []
    for s in load(ROOT):
        shard, row = where[s.index]
        H, W = s.image.shape[:2]
        fr = frame_row(NAME, shard.stem, row, "test", rel(shard), np.nan, W, H)
        frames.append(fr)
        for k, (name, box) in enumerate(sorted(s.boxes.items(), key=lambda kv: TRACK[kv[0]])):
            people.append({"sample": fr["sample"], "person": k, "track": TRACK[name], "box": box.tolist(),
                           "kp": empty_kp_list(), "stroke": None, "labeler": NAME})
        if s.ball is not None:
            ball.append({"sample": fr["sample"], "x": s.ball[0], "y": s.ball[1], "visible": True, "labeler": NAME})
    return {"frames": pd.DataFrame(frames), "people": pd.DataFrame(people), "ball": pd.DataFrame(ball)}


if __name__ == "__main__":
    run(NAME)
