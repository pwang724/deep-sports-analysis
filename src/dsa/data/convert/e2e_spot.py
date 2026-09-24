"""E2E-Spot tennis -> events, spans.

data/tennis/{train,val,test}.json: clips "<match>_<start>_<end>" with fps and
events {frame (from clip start), label <near|far>_court_<serve|swing|bounce>,
comment}. Stored in whole-video frames (start + frame) of
videos/e2e_spot/<match>.mp4, at the clip's fps as labelled (29.97 for US
Open, 25 for Wimbledon; the per-clip values differ in the 4th decimal).
swing -> shot; side is the court half of the hitter or of the bounce.

Only test clips carry stroke comments; train and val say "extended dataset".
forehand_ / backhand_ + topspin / slice / volley give hand and technique
(topspin -> gs); overhead -> smash with no hand; special, serve, bounce_time
and blank add nothing. Each clip is a span covering "serve,shot,bounce"
(end = the clip name's end, exclusive: end - start = num_frames). No frames
table: nothing here is per-frame. Two overlapping test clips of the 2019 Wimbledon final
repeat two shots; exact duplicates are dropped. All 28 matches are
downloaded whole at 720p and the labelled rate by `python -m dsa.data.broadcast`.
"""
from __future__ import annotations

import json

import pandas as pd

from dsa.data.convert import events_frame, rel, run, span
from dsa.data.paths import SOURCES, VIDEOS

NAME = "e2e_spot"
ROOT = SOURCES / "e2e_spot/data/tennis"
TYPE = {"serve": "serve", "swing": "shot", "bounce": "bounce"}
TECH = {"topspin": "gs", "slice": "slice", "volley": "volley"}


def stroke(comment: str) -> tuple[str | None, str | None]:
    if comment == "overhead":
        return None, "smash"
    hand, _, tech = comment.partition("_")
    if hand in ("forehand", "backhand") and tech in TECH:
        return hand, TECH[tech]
    return None, None


def convert() -> dict:
    events, spans = [], []
    for split in ("train", "val", "test"):
        for clip in json.load(open(ROOT / f"{split}.json")):
            match, start, end = clip["video"].rsplit("_", 2)
            media = rel(VIDEOS / NAME / f"{match}.mp4")
            spans.append(span(NAME, split, media, int(start), int(end), clip["fps"], "serve,shot,bounce"))
            for e in clip["events"]:
                side, _, kind = e["label"].split("_")
                hand, tech = stroke(e["comment"])
                events.append({"source": NAME, "split": split, "media": media, "frame": int(start) + e["frame"],
                               "fps": clip["fps"], "type": TYPE[kind], "side": side, "hand": hand, "technique": tech,
                               "labeler": NAME})
    return {"events": events_frame(events).drop_duplicates(ignore_index=True), "spans": pd.DataFrame(spans)}


if __name__ == "__main__":
    run(NAME)
