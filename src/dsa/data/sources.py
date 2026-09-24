"""Every public dataset we use: where it lives, where it came from, what it labels.

`python -m dsa.data.sources` prints the table with what is on disk.
"""
from __future__ import annotations

from dataclasses import dataclass

from dsa.data.paths import LABELS, SOURCES


@dataclass(frozen=True)
class Source:
    name: str             # directory under data/sources/ and data/labels/
    url: str
    license: str
    media: str            # what the frames are
    labels: str           # which heads it supervises, in schema terms
    split: str            # how train / val / test are assigned
    note: str = ""


REGISTRY = [
    Source("tracknet", "https://drive.google.com/file/d/1lhAaeQCmk2y440PmagA0KmIVBIysVMwu (TrackNet tennis)",
           "research use", "10 broadcast games, 95 clips, 19,835 1280 x 720 JPEG frames, 30 fps",
           "ball (x, y, visible), events (shot, bounce)", "games 1-7 train, 8-10 test (WASB's split)"),
    Source("tennis_court_detector", "https://github.com/yastrebksv/TennisCourtDetector", "MIT",
           "8,841 broadcast frames, 1280 x 720 PNG", "court (14 points)",
           "data_train.json train, data_val.json val",
           "only the labels and 100 val images are unpacked; the zip holds the rest"),
    Source("racketvision", "https://huggingface.co/datasets/linfeng302/RacketVision", "MIT (labels)",
           "YouTube broadcast rallies, 1080p MP4, 20% of frames labelled",
           "rackets (box + 5 points), ball (x, y, visible)", "its info/train|val|test.json by (match, rally)",
           "tennis subset: 431 matches, one 10 s rally each; labels for all 431 on disk, videos only for "
           "matches 1, 10, 100 locally, all 431 on the Modal volume (datasets/racketvision); fps varies by "
           "match (279 at 25, 32 at 30, 1 at 50, 119 at 60)"),
    Source("coco_wholebody", "https://github.com/jin-s13/COCO-WholeBody", "CC BY-NC 4.0 (labels)",
           "COCO 2017 photos, not tennis", "people (box, 17 body + 6 feet)", "val2017 val",
           "val annotations only; images fetched one by one as used"),
    Source("tennis_segmentation", "https://huggingface.co/datasets/julia-wenkmann/TennisSegmentation",
           "none stated: evaluation only",
           "197 broadcast end-on frames, 1280 x 720 (108) or 1920 x 1080 (89), inside Parquet shards",
           "people (box, both players), ball (mask centre)", "all test"),
    Source("tennis_player_actions", "https://data.mendeley.com/datasets/nv3gv6d9sh (Mendeley)", "CC BY 4.0",
           "~2,000 self-recorded images, one player each", "people (box, 17 body + neck), stroke (4 classes)",
           "all test (phase-0 scoring set)"),
    Source("e2e_spot", "https://github.com/jhong93/spot", "BSD-3 (labels); videos on YouTube",
           "US Open (29.97 fps) / Wimbledon (25 fps) broadcasts by YouTube id, 1080p",
           "events (serve, shot, bounce; side; coarse stroke)", "its train / val / test.json",
           "only the US Open 2019 final is downloaded (data/videos/broadcast); stroke comments on test only"),
    Source("f3set", "https://github.com/F3Set/F3Set", "see repo; videos on YouTube",
           "151 broadcast matches by YouTube id, 720p, mostly 25 or 29.97 fps (videos.csv)",
           "events (serve, shot; side, hand, technique, direction, outcome)", "its train / val / test.json",
           "only two test matches are downloaded (data/videos/broadcast)"),
]
BY_NAME = {s.name: s for s in REGISTRY}


def main() -> None:
    for s in REGISTRY:
        on_disk = "yes" if (SOURCES / s.name).exists() else "no"
        converted = "yes" if any((LABELS / s.name).glob("*.parquet")) else "no"
        print(f"{s.name:24} on disk: {on_disk:3}  converted: {converted:3}  {s.labels}")


if __name__ == "__main__":
    main()
