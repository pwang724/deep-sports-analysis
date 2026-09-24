"""Sample the gold set: frames spread across every kind of footage we hold.

Draws from three collections, one frame per sampled moment:

  youtube    one frame per downloaded segment (data/videos/youtube/manifest.csv),
             so each video gives up to 3 frames and no video dominates.
  broadcast  the E2E-Spot / F3Set test matches in data/videos/broadcast.
  own        our recordings in data/raw/tennis_videos.

Frames are taken at random times away from segment edges. The videos used
are the gold set's and must never enter training: `videos.txt` lists them.

    python -m dsa.label.gold --version v1 --broadcast 10 --own 20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from dsa.data import schema
from dsa.data.paths import DATA, GOLD, RAW, VIDEOS


def probe(path: Path) -> tuple[float, int, int, int]:
    cap = cv2.VideoCapture(str(path))
    out = (cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
           int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    return out


def moments(path: Path, n: int, rng: np.random.Generator, edge_s: float = 2.0) -> list[dict]:
    fps, count, w, h = probe(path)
    lo, hi = int(edge_s * fps), count - int(edge_s * fps) - 1
    if hi <= lo or n <= 0:
        return []
    frames = sorted(rng.choice(np.arange(lo, hi), min(n, hi - lo), replace=False))
    return [{"media": str(path.relative_to(DATA)), "frame": int(f), "fps": fps, "width": w, "height": h}
            for f in frames]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--version", default="v1")
    p.add_argument("--per-segment", type=int, default=1)
    p.add_argument("--broadcast", type=int, default=10, help="frames per broadcast match")
    p.add_argument("--own", type=int, default=20, help="frames per own recording")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng = np.random.default_rng(a.seed)
    source = f"gold_{a.version}"

    rows = []
    manifest = VIDEOS / "youtube" / "manifest.csv"
    if manifest.exists():
        m = pd.read_csv(manifest)
        m = m[m.get("selected", True).fillna(False).astype(bool) & ~m["status"].fillna("").str.contains("reject")]
        for _, v in m.iterrows():
            for seg in sorted((VIDEOS / "youtube" / v.video_id).glob("seg*.mp4")):
                if seg.name.startswith("._"):
                    continue
                for r in moments(seg, a.per_segment, rng):
                    rows.append({**r, "collection": "youtube", "video": v.video_id,
                                 **{k: v.get(k) for k in ("category", "camera", "surface", "level", "play")}})
    for path in sorted((VIDEOS / "broadcast").glob("*.mp4")):
        if not path.name.startswith("._"):
            rows += [{**r, "collection": "broadcast", "video": path.stem, "camera": "broadcast", "level": "pro"}
                     for r in moments(path, a.broadcast, rng)]
    for path in sorted((RAW / "tennis_videos").glob("*.MOV")):
        if not path.name.startswith("._"):
            rows += [{**r, "collection": "own", "video": path.stem, "camera": "baseline_low", "level": "rec"}
                     for r in moments(path, a.own, rng)]

    df = pd.DataFrame(rows)
    df["source"] = source
    df["split"] = "test"
    df["sample"] = [schema.sample_id(source, Path(m).with_suffix("").as_posix().replace("/", "__"), f)
                    for m, f in zip(df.media, df.frame)]
    out = GOLD / a.version
    out.mkdir(parents=True, exist_ok=True)
    schema.validate("frames", df)
    df.to_parquet(out / "frames.parquet", index=False)
    (out / "videos.txt").write_text("\n".join(sorted(df.video.unique())) + "\n")
    print(df.groupby("collection").agg(frames=("sample", "size"), videos=("video", "nunique")).to_string())
    for col in ("camera", "surface", "level"):
        if col in df:
            print(f"\n{col}: " + ", ".join(f"{k} {v}" for k, v in df[col].fillna("unknown").value_counts().items()))
    print(f"\n-> {out / 'frames.parquet'}")


if __name__ == "__main__":
    main()
