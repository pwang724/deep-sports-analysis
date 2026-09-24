"""F3Set tennis -> events, spans.

data/f3set-tennis/{train,val,test}.json: clips "<match>_<start>_<end>", one
event per shot (no bounces), label "<side>_<position>_<serve|return|stroke>_
<fh|bh|->_<technique|->_<direction>_<approach|->_<outcome>" (tokens in
elements.txt). Stored in whole-video frames (start + frame) of
videos/broadcast/<match>.mp4. serve -> type serve, return and stroke -> shot;
fh / bh -> forehand / backhand; technique, direction (T / B / W on serves) and
outcome (in, winner, forced-err, unforced-err) as given. Court position
(deuce / middle / ad) and the approach flag have no schema field and are
dropped.

fps: the clip json truncates it (29.0 for 29.97), so it comes from
videos.csv, which gives 25, 29.97, 24.975, 23.976 or 30 per match; three
matches missing from videos.csv (two Roland Garros 2015, Madrid 2019) take
the json value, a truncated 23 / 24 / 29 read as the NTSC rate (x 1000 /
1001). Frame numbers are at that fps even where the downloaded video differs
(the Jabeur match is 59.94 fps on disk). Some matches appear in both
train and val (the source splits by clip). Each clip is a span covering
"serve,shot" (no bounces). No frames table (see e2e_spot).
"""
from __future__ import annotations

import csv
import json

import pandas as pd

from dsa.data.convert import events_frame, rel, run, span
from dsa.data.paths import SOURCES, VIDEOS

NAME = "f3set"
ROOT = SOURCES / "f3set/data/f3set-tennis"
HAND = {"fh": "forehand", "bh": "backhand"}
NTSC = {23.0: 24000 / 1001, 24.0: 25000 / 1001, 29.0: 30000 / 1001}


def convert() -> dict:
    fps = {r["video_name"]: float(r["fps"]) for r in csv.DictReader(open(ROOT / "videos.csv"))}
    events, spans = [], []
    for split in ("train", "val", "test"):
        for clip in json.load(open(ROOT / f"{split}.json")):
            match, start, end = clip["video"].rsplit("_", 2)
            media = rel(VIDEOS / "broadcast" / f"{match}.mp4")
            rate = fps.get(match) or NTSC.get(clip["fps"], clip["fps"])
            spans.append(span(NAME, split, media, int(start), int(end), rate, "serve,shot"))
            for e in clip["events"]:
                side, _, kind, hand, tech, direction, _, outcome = e["label"].split("_")
                events.append({"source": NAME, "split": split, "media": media, "frame": int(start) + e["frame"],
                               "fps": rate, "type": "serve" if kind == "serve" else "shot",
                               "side": side, "hand": HAND.get(hand), "technique": None if tech == "-" else tech,
                               "direction": direction, "outcome": outcome, "labeler": NAME})
    return {"events": events_frame(events), "spans": pd.DataFrame(spans)}


if __name__ == "__main__":
    run(NAME)
